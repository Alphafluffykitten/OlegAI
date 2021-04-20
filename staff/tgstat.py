from staff.sqlite_db import SQLiteDB
import json,time
from tqdm import trange
import threading
from random import randint
import logging

class TGStatDB:
    """ methdos for TGStat DB """
    
    sleep_counter = 0       # counts sleep cycles to know when to do long sleep
    fast_sleep = (30,50)    # sleep between tasks
    long_sleep = 600        # long sleep, seconds
    long_sleep_run = 20     # how many task runs to fall into long sleep
    super_sleep = 3600      # if caught rate limiting
    brake = False           # stop current task
    rate_limit = False
    running = False

    def __init__(self,db_file,admin):
        self.db_file = db_file
        self.admin = admin

        # set logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('tgstat_logs/tgstat_log.txt')
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(funcName)s:\n %(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def start(self):
        self.db = SQLiteDB(self.db_file)
        threading.Thread(target=self.join_chat_list).start()
        
    def stop(self):
        self.brake = True
        while self.running:
            time.sleep(0.1)
        self.db.close()
        
    def from_file_to_db(self,file):
        ''' reads json from file and writes all info to DB '''
        
        with open(file) as f:
            tgstat_json = json.loads(f.read())
            
        count = 0
        for ch in tgstat_json:
            in_db = self.get_channels(id = ch['id'])
            
            if not in_db:
                self.add_channel(ch)
                count += 1
                print (f'{count} written')
            else:
                print(f"channel id {ch['id']} {ch['username']} already in database")
        
        return count
    

    def get_column_names(self,table):
        sql = f'pragma table_info({table})'
        rows = self.db.query(sql)
        columns = []
        for r in rows:
            columns.append(r[1])
        return columns

                
    def get_channels(self,
                     id=None,
                     has_username=None,
                     chat_id_null = None,
                     chat_id_not_null = None,
                     chat_id_failed = None,
                     joined = None
                    ):
        clauses = []
        if has_username is not None:
            if has_username == True:
                clauses.append(f"NOT username = ''")
            else:
                clauses.append(f"username = ''")
        if id is not None:
            clauses.append(f'id = {id}')
        if chat_id_null:
            clauses.append('chat_id IS NULL')
        if chat_id_not_null:
            clauses.append('chat_id NOT NULL')
        if chat_id_failed is not None:
            clauses.append(f'chat_id_failed = {int(chat_id_failed)}')
        if joined is not None:
            clauses.append(f'joined = {int(joined)}')
        
        if clauses:
            clauses = ' AND\n'.join(clauses)
            clauses = 'WHERE ' + clauses
            
        cols = self.get_column_names('tgstat_4500')
            
        sql = f'''
            SELECT {', '.join(cols)}
            FROM tgstat_4500 
            {clauses}
            ORDER BY id ASC
        '''
        #print (sql)
        rows = self.db.query(sql)
        
        res = []
        for r in rows:
            res.append({colname:r[idx] for idx,colname in enumerate(cols)})
        return res
        
    def add_channel(self,ch):
        sql = f'''
            INSERT INTO tgstat_4500 (id, tg_id, link, username, title, about, participants_count, tgstat_restrictions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ? )
        '''
        #print(sql)
        self.db.push(sql,[ch['id'], ch['tg_id'], ch['link'], ch['username'], ch['title'], 
                          ch['about'], ch['participants_count'], dumps(ch['tgstat_restrictions'])
                         ]
                    )
        
    def add_chat_id_from_file(self,file):
        with open(file) as f:
            tgstat_json = json.loads(f.read())
            
        count = 0
        for ch in tgstat_json:
            if 'chat_id' in ch:
                sql = f'''
                UPDATE tgstat_4500
                SET chat_id = {ch['chat_id']}
                WHERE id = {ch['id']}
                '''
                self.db.push(sql)
                count +=1
                #print(f'{count } chat ids written')
                
    def update_channel(self, id, chat_id=None, chat_id_failed=None, joined=None, error_code=None, error_info=None):
        clauses = []
        if chat_id:
            clauses.append(f'chat_id = {chat_id}')
        if chat_id_failed is not None:
            clauses.append(f'chat_id_failed = {int(chat_id_failed)}')
        if joined is not None:
            clauses.append(f'joined = {int(joined)}')
        if error_code is not None:
            clauses.append(f'error_code = {error_code}')
        if error_info is not None:
            clauses.append(f'error_info = "{error_info}"')

        if clauses:
            clauses = ',\n'.join(clauses)
            sql = f'''
            UPDATE tgstat_4500
            SET
            {clauses}
            WHERE
            id = {id}
            '''
            self.db.push(sql)

    def join_chat_list(self):
        """
        takes all channels that are not joined and chat_id_failed=0 and tries to join them
        either by invite link or by obtaining chat_id via username
        """
        self.running = True
        channels = self.get_channels(joined = False, chat_id_failed=0)

        self.logger.info(f'Got {len(channels)} channels')

        for c in channels:

            # for channels with username: get chat_id and join chat by it
            if c['username']:
                chat_id, _ = self.get_chat_id(c)
                self.sleep()
                if chat_id:
                    self.join_username(c, chat_id)

            # for channels with link: just try to join with link
            elif 'joinchat' in c['link']:
                self.join_link(c)

            self.sleep()
            if self.brake:
                self.brake = False
                break

        self.running = False

    def get_chat_id(self, c):
        res = self.admin.tdutil.search_public_chat(tg_username = c['username'])
        if not res.error:
            if res.update.get('@type','') == 'chat':
                chat_id = res.update.get('id', None)
                if chat_id:
                    self.logger.info(f"Got {c['username']} = {chat_id}")
                    return chat_id, res
                else:
                    return False, res
            else:
                self.logger.warning(f'Got unexpected update:\n{res.update}')
        else:
            self.update_channel(
                id = c['id'], chat_id_failed = True,
                error_code = res.error_info['code'],
                error_info = res.error_info['message']
            )
            self.logger.error(f"Channel {c['username']} caused error\n {res.error_info}")
            if res.error_info['code'] == 429:
                self.rate_limit_caught()

            return False, res

    def join_username(self, c, chat_id):
        joined, res = self.admin.join_chat_mute(chat_id = chat_id)
        if res.error:
            self.logger.error(f"Channel {c['username']} caused error:\n{res.error_info}")
            self.update_channel(id = c['id'], error_code = res.error_info['code'], error_info = res.error_info['message'])
            if res.error_info['code'] == 429:
                self.rate_limit_caught()

        if joined:
            self.update_channel(id=c['id'], joined = True, chat_id = chat_id)
            self.logger.info(f"channel {c['chat_id']} | {c['username']} joined")

    def join_link(self, c):
        joined, res = self.admin.join_chat_mute(link=c['link'])
        if res.error:
            self.update_channel(id = c['id'], error_code = res.error_info['code'], error_info = res.error_info['message'])
            self.logger.error(f"Channel {c['title']} caused error:\n{res.error_info}")
            if res.error_info['code'] == 429:
                self.rate_limit_caught()

        if joined:
            self.update_channel(id=c['id'], joined = True)
            self.logger.info(f"channel {c['title']} | link joined")

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



