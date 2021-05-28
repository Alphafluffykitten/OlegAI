from staff.pgsql import PostgresDB
from staff.olegtypes import ChannelPool
import json, time
import threading
from random import randint
import logging
import os
import requests

class CrawlerDB:
    """ methods to work with DB data in channel_pool """
    
    def __init__(
        self,
        db_name,
        db_user,
        db_password,
        db_host):

        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host

    def start(self):
        self.db = PostgresDB(self.db_name, self.db_user, self.db_password, self.db_host)
        
    def stop(self):
        self.db.close()
                
    def get_channels(
        self,
        id = None,
        tgstat_id = None,
        has_username = None,
        username = None,
        link = None,
        chat_id_null = None,
        chat_id_not_null = None,
        chat_id_failed = None,
        joined = None):
        """
        Gets channels from TGStatDB
        """

        clauses = []
        if has_username is not None:
            if has_username == True:
                clauses.append(f"NOT username = ''")
            else:
                clauses.append(f"username = ''")
        if username is not None:
            clauses.append(f"username = '{username}'")
        if link is not None:
            clauses.append(f"link = '{link}'")
        if id is not None:
            clauses.append(f'id = {id}')
        if tgstat_id is not None:
            clauses.append(f'tgstat_id = {tgstat_id}')
        if chat_id_null:
            clauses.append('chat_id IS NULL')
        if chat_id_not_null:
            clauses.append('NOT chat_id IS NULL')
        if chat_id_failed is not None:
            clauses.append(f'chat_id_failed = {bool(chat_id_failed)}')
        if joined is not None:
            clauses.append(f'joined = {bool(joined)}')
        
        if clauses:
            clauses = ' AND\n'.join(clauses)
            clauses = 'WHERE ' + clauses
            
        sql = f'''
            SELECT {', '.join(ChannelPool.cols)}
            FROM {ChannelPool.table_name} 
            {clauses}
            ORDER BY RANDOM()
        '''
        #print (sql)
        rows = self.db.query(sql)
        
        res = []
        for r in rows:
            res.append({colname:r[idx] for idx,colname in enumerate(ChannelPool.cols)})
        return res
    
    def update_channel(self, **kwargs):
        """
        Updates channel if exists, inserts new if not
        """
        upd = {}
        for k in kwargs:
            if k in ChannelPool.cols:
                upd[k] = kwargs[k]

        in_db = None
        if upd.get('id', None) is not None:
            in_db = self.get_channels(id = upd['id'])
        elif upd.get('tgstat_id', None) is not None:
            in_db = self.get_channels(tgstat_id = upd['tgstat_id'])
        elif upd.get('username', None) is not None:
            in_db = self.get_channels(username = upd['username'])
        elif upd.get('link', None) is not None:
            in_db = self.get_channels(link = upd['link'])
        else:
            return

        if in_db and upd:
            clauses = []
            for a in upd:
                clauses.append(f'{a} = %s')
            clauses = ',\n'.join(clauses)
            sql = f'''
            UPDATE {ChannelPool.table_name}
            SET
            {clauses}
            WHERE
            id = {in_db[0]['id']}
            '''
        elif upd:
            ss = ["%s" for i in range(len(upd))]
            sql = f'''
            INSERT INTO {ChannelPool.table_name} ({', '.join(list(upd.keys()))})
            VALUES ({', '.join(ss)})
            '''
        else:
            return

        self.db.push(sql,list(upd.values()))


class Crawler():
    """
    Base class for crawling TG channels and making requests with timeouts.
    Rate limit aware
    """

    sleep_counter = 0       # counts sleep cycles to know when to do long sleep
    fast_sleep = (30,50)    # sleep between tasks
    long_sleep = 600        # long sleep, seconds
    long_sleep_run = 20     # how many task runs to fall into long sleep
    super_sleep = 3600      # if caught rate limiting
    brake = False           # stop current task
    rate_limit = False
    running = False

    def rate_limit_caught(self):
        self.logger.warning(f'Rate limit caught, sleeping {self.super_sleep/60} min')
        self.rate_limit = True
        self.sleep()
        self.rate_limit = False
            
    def sleep(self):
        i = self.sleep_counter
        dur = randint(*self.fast_sleep)
        if (i+1) % self.long_sleep_run == 0:
            self.logger.info(f'Long sleep {self.long_sleep/60} min')
            dur = self.long_sleep
        if self.rate_limit:
            dur = self.super_sleep
        for j in range(dur):
            if self.brake == True: return
            time.sleep(1)
        self.sleep_counter += 1

# TODO: make join_chat_list get each new channel from DB one by one.
# If we make spider that adds new stuff to pool, it'll be handy. 
# If no channels are available in the pool, wait say 10 sec and repeat.
class Joiner(Crawler):
    """
    joins channels from channel_pool
    """

    def __init__(self, dba, lhub, logs_dir):
        self.dba = dba
        self.lhub = lhub

        # set up logging
        joiner_logs = os.path.join(logs_dir,'joiner')
        if 'joiner' not in os.listdir(logs_dir):
            os.mkdir(joiner_logs)
        logger = logging.getLogger('Joiner')
        logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{joiner_logs}/{len(os.listdir(joiner_logs))}.txt')
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter(
            '%(asctime)s %(name)s [%(levelname)s] %(funcName)s:\n %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        self.logger = logger


    def start(self):
        if not self.running:
            self.logger.info('Starting Joiner')
            threading.Thread(target=self.join_chat_list).start()
        else:
            self.logger.info('Joiner already running')

    def stop(self):
        if self.running:
            self.logger.info('Waiting for current task to end')
            self.brake = True
            while self.running:
                time.sleep(0.1)
            self.logger.info('Joiner stopped')

    def join_chat_list(self):
        """
        takes all channels that are not joined and chat_id_failed=0 and tries to join them
        either by invite link or by obtaining chat_id via username
        """
        self.running = True
        channels = self.dba.get_channels(joined = False, chat_id_failed = False)

        self.logger.info(f'Got {len(channels)} channels')

        for c in channels:

            # for channels with username: get chat_id and join chat by it
            if c['username']:
                chat_id, _ = self.get_chat_id(c)
                c['chat_id'] = chat_id
                self.sleep()
                if chat_id:
                    self.join(by='chat_id', c=c)

            # for channels with link: just try to join with link
            elif 'joinchat' in c['link']:
                self.join(by='link', c=c)

            self.sleep()
            if self.brake:
                self.brake = False
                break

        self.running = False

    def get_chat_id(self, c):
        """ Gets telegram chat_id by username """

        listener = self.lhub.get_min_listener()
        res = listener.tdutil.search_public_chat(tg_username = c['username'])
        if not res.error:
            if res.update.get('@type','') == 'chat':
                chat_id = res.update.get('id', None)
                if chat_id:
                    self.logger.info(f"Got chat_id {c['username']} = {chat_id}")
                    return chat_id, res
                else:
                    return False, res
            else:
                self.logger.warning(f'Got unexpected update from TDLib:\n{res.update}')
        else:
            self.logger.error(f"Channel {c['username']} caused error\n {res.error_info}")
            if res.error_info['code'] == 429:
                self.rate_limit_caught()
                return False, res
            
            self.dba.update_channel(
                id = c['id'],
                chat_id_failed = True,
                error_code = res.error_info['code'],
                error_info = res.error_info['message']
            )

            return False, res

    def join(self, by, c):
        """
        Joins channel and writes result to CrawlerDB
        Args:
        by (str): can be 'chat_id' or 'link'
        с (dict): channel from CrawlerDB
        """

        if by == 'chat_id':
            joined, res, channel = self.lhub.join_chat_mute(chat_id = c['chat_id'])
        elif by == 'link':
            joined, res, channel = self.lhub.join_chat_mute(link = c['link'])
        else:
            raise Exception('[ Crawler.join ]: by can be either "chat_id" or "link"')

        if getattr(res, 'error', False):
            self.logger.error(f"Channel {c['username']} {c['title']} caused error:\n{res.error_info}")
            self.dba.update_channel(
                id = c['id'], 
                error_code = res.error_info['code'], 
                error_info = res.error_info['message'])
            if res.error_info['code'] == 429:
                self.rate_limit_caught()
            elif res.error_info['code'] == 1001:
                self.logger.warning('Maximum number of channels for all listeners reached')
                self.brake = True

        if joined:
            self.dba.update_channel(
                id=c['id'], 
                joined = True, 
                chat_id = channel.tg_channel_id, 
                internal_channel_id = channel.id)
            self.logger.info(f"Channel {c['chat_id']} | {c['username']} {c['title']} joined")



class TGStat():
    """ tgstat.ru API """

    def __init__(self, token, dba):
        self.dba = dba
        self.token = token
    
    def get_top_channels(self):

        codes = self.get_cat_codes()
        print(codes)

        list_of_lists = []
        for cat in codes:
            channels = self.get_cat_channels(cat)
            list_of_lists.append(channels)
            
        one_list = []
        for l in list_of_lists:
            for i in l:
                one_list.append(i)

        return one_list


    def get_cat_channels(self, cat):
        """ get channels from category """

        payload = {'token': self.token, 'category': cat, 'country': 'ru', 'limit': 100}
        r = requests.get('https://api.tgstat.ru/channels/search', params=payload)
        channels = json.loads(r.text)
        channels = channels['response']['items']

        return channels

    def get_cat_codes(self):
        """ get TGstat categories codes """

        payload = {'token': self.token}
        r = requests.get('https://api.tgstat.ru/database/categories', params=payload)
        
        cat_json = json.loads(r.text)
        
        codes = []
        for c in cat_json['response']:
            codes.append(c['code'])
            
        return codes

    def save_json(self,file, obj):

        to_file = json.dumps(obj,indent=4,ensure_ascii=False)
        with open(file, 'w') as f:
            f.write(to_file)

    def load_json(self, file):
        with open(file) as f:
            obj = json.loads(f.read())

        return obj

    def save_to_db(self, channels):
        """ updates info on channels in CrawlerDB """

        for ch in channels:
            self.dba.update_channel(
                tgstat_id=ch['id'],
                link = ch['link'],
                username = ch['username'],
                title = ch['title'],
                about = ch['about'],
                chat_id_failed = False
            )