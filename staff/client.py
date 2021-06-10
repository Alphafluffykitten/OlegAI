from staff.olegtypes import *
from staff.dba import OlegDBAdapter
from staff.utils import Converters
from staff.utils import OlegHashing
from staff.utils import TDLibUtils
from staff.nn import OlegNN
from staff.crawler import Joiner, CrawlerDB
from staff.filters import Filters
from apscheduler.schedulers.background import BackgroundScheduler
import os
import logging
from types import SimpleNamespace
import time, datetime


class OlegApp():
    """ superclass composing all inhabitants of the app """
    
    def __init__(self,
                 db_name,
                 db_user,
                 db_password,
                 db_host,
                 bot_token,
                 api_id,
                 api_hash,
                 listener_db_encryption_key,
                 bot_db_encryption_key,
                 listener_files_directories,
                 bot_files_directory,
                 salt,
                 logs_dir,
                 admin_user_id,
                 nn_user_lpv_len,
                 nn_post_lpv_len,
                 nn_channel_lpv_len,
                 nn_hidden_layer,
                 nn_new_reactions_threshold,
                 nn_learning_timeout,
                 nn_full_learn_threshold,
                 nn_closest_shuffle,
                 max_listener_channels,):

        self.logs_dir = logs_dir
        self._setup_logs_dir()

        self.logger = self._setup_logging()
        
        self.dba = OlegDBAdapter(db_name,db_user,db_password,db_host)
        self.conv = Converters()
        self.hash = OlegHashing(salt = salt, trunc = 7)
        self.nn = OlegNN(
            nn_user_lpv_len,
            nn_post_lpv_len,
            nn_channel_lpv_len,
            nn_hidden_layer,
            nn_new_reactions_threshold,
            nn_learning_timeout,
            nn_full_learn_threshold,
            nn_closest_shuffle,)

        self.lhub = ListenerHub(
            api_id = api_id,
            api_hash = api_hash,
            database_encryption_key = listener_db_encryption_key,
            files_directory = listener_files_directories,
            max_listener_channels = int(max_listener_channels))
        
        self.bot = Bot(
            api_id = api_id,
            api_hash = api_hash,
            bot_token = bot_token,
            database_encryption_key = bot_db_encryption_key,
            files_directory = bot_files_directory,
            admin_user_id = admin_user_id,)

        self.crawler_db = CrawlerDB(db_name,db_user,db_password,db_host)
        self.joiner = Joiner(self.crawler_db, self.lhub, logs_dir = logs_dir)
        
        self.scheduler = BackgroundScheduler(timezone = 'Europe/Moscow')
        self.scheduled = self.scheduler.add_job(
            func = self.bot.scheduled_mailing,
            trigger = 'cron',
            hour = '10,12,14,16,19',
            minute = '20',
            coalesce = True,)

        self.debug = SimpleNamespace()

        # attach OlegApp to sub-objects so they have access to other inhabitants of app
        self.bot.app = self
        self.lhub.app = self
        self.nn.app = self
        self.conv.app = self
        self.dba.app = self

    def start(self):
        """ should be called before anything """
        
        self.dba.start()
        self.nn.start()
        self.lhub.start()
        self.bot.start()
        self.scheduler.start()
        self.crawler_db.start()
        
    def stop(self):
        """ graceful shutdown """
        
        self.joiner.stop()
        self.crawler_db.stop()
        self.scheduler.shutdown(wait=False)
        self.bot.stop()
        self.lhub.stop()
        self.dba.stop()

    def _setup_logs_dir(self):
        if not os.path.isdir(self.logs_dir):
            os.mkdir(self.logs_dir)

    def _setup_logging(self):
        
        app_logs = os.path.join(self.logs_dir,'app')
        if 'app' not in os.listdir(self.logs_dir):
            os.mkdir(app_logs)
        logger = logging.getLogger('OlegApp')
        logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{app_logs}/{len(os.listdir(app_logs))}.txt')
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
        return logger







class Bot():
    """ 
    implements interaction functionality between
    bot and bot user
    """
    
    def __init__(
        self,
        api_id,
        api_hash,
        bot_token,
        database_encryption_key,
        files_directory,
        admin_user_id,
    ):
        
        self.tdutil = TDLibUtils(
            api_id = api_id,
            api_hash = api_hash,
            database_encryption_key = database_encryption_key,
            bot_token = bot_token,
            files_directory = files_directory)

        self.admin_user_id = admin_user_id
        
    def start(self):
        
        # this executes upon receiving any update.message from TDLib
        self.tdutil.greeting = self.user_greeting

        # set bot handlers
        #self.tdutil.add_command_handler('start',self.start_handler)
        self.tdutil.add_command_handler('send_new',self.send_new_handler)
        self.tdutil.add_command_handler('send_channel',self.send_channels_last)
        self.tdutil.add_command_handler('start_joiner', self.start_joiner_handler, Filters.user(self.admin_user_id))
        self.tdutil.add_command_handler('stop_joiner', self.stop_joiner_handler, Filters.user(self.admin_user_id))
        self.tdutil.tg.add_update_handler('updateNewCallbackQuery', self.callback_query_handler)

        # set got_admin_forward handler for every listener
        for l in self.app.lhub.ls:
            self.tdutil.add_message_handler(
                Filters.forwarded & Filters.user(self.app.lhub.ls[l].user_id),
                self.got_admin_forward
            )

        self.tdutil.start()

        self.user_id = self.tdutil.get_me()
        self.reactions = self.app.dba.get_reactions()

        
    def stop(self):
        self.tdutil.stop()
        
    def user_greeting(self, message):
        """ if user is unknown, register him """

        # if message from a user, not a chat
        if  message.get('sender',{}).get('@type','') == 'messageSenderUser':
            tg_user_id = message.get('chat_id',0)
            username = self.tdutil.get_username(tg_user_id)
            # if there is no such user in OlegDB
            if not self.app.dba.get_user(tg_user_id = tg_user_id):
                user = self.app.dba.register_user(tg_user_id, username)
                if user:
                    self.app.nn.handle_new_obj('user', user)
                    self.user_bootstrap(user)
                    self._users_send_new([user])
                    
    def user_bootstrap(self, user):
        """ routine for newly registered user """

        res = self.tdutil.tg.send_message(chat_id=user.tg_user_id,
            text='Я собираю посты со всех русскоязычных пабликов Телеграма'
        )
        res.wait()
        time.sleep(3)
        self.tdutil.tg.send_message(chat_id=user.tg_user_id,
            text='Лучшие посты буду отправлять вам'
        )
        res.wait()
        time.sleep(3)
        self.tdutil.tg.send_message(chat_id=user.tg_user_id,
            text='Ставьте 👍 или 💩, я обучаюсь на ваших оценках'
        )
        res.wait()
        time.sleep(3)
        
    def start_handler(self, message, params):
        """ on /start """
        
        pass

        #tg_user_id = message.get('chat_id',0)
        #self.tdutil.tg.send_message(chat_id=tg_user_id, text='Hi!')
        
    
    def send_new_handler(self, message, params):
        """ sends new post when user asks with /send_new """
        
        tg_user_id = message.get('chat_id',0)
        user = self.app.dba.get_user(tg_user_id=tg_user_id)
        self._users_send_new([user])

    def send_channels_last(self, message, params):
        """ sends last post from given channel to user from which message was received """

        tg_user_id = message.get('chat_id',0)
        user = self.app.dba.get_user(tg_user_id = tg_user_id)
        post = self.app.dba.get_posts(tg_channel_id = params[0], have_content=True, limit=1)
        if post:
            post = post[0]

            to_send = self._prepare_message(user,post)

            res = self.tdutil.send_message(to_send)
            if res.error:
                self.app.logger.error(res.error_info)
        else:
            res = self.tdutil.tg.send_message(chat_id=user.tg_user_id, text=f'No posts found from channel {params[0]}')

    def start_joiner_handler(self,message, params):
        self.app.joiner.start()

    def stop_joiner_handler(self, message, params):
        self.app.joiner.stop()

    def _users_send_new(self, users):
        """ For each user calls nn.closest and sends best post """

        posts = self.app.nn.posts_cache.posts
        
        if not posts:
            #self.tdutil.tg.send_message(chat_id=user.tg_user_id, text='No new messages :(')
            self.app.logger.warning('[ Bot._users_send_new ]: No messages found')
            return

        for u in users:
            to_send = None
            for i in range(100):
                filtered = self.app.nn.closest(posts, u, offset=i)
                # try to find post that havent been reposted
                if not self.app.dba.get_reposts(filtered.id, u.id):
                    to_send = self._prepare_message(u, filtered)
                    # check if it's still there
                    if to_send:
                        break
            if to_send: 
                res = self.tdutil.send_message(to_send)
                if res.error:
                    self.app.logger.error('[ Bot._users_send_new ]: '+str(res.error_info))

                # write reposted info to OlegDB.reposts
                self.app.dba.add_repost(filtered, u)


    def _prepare_message(self,user,post):
        """
        Prepares tdmessage for sending (name, converters etc)

        Returns: if message was found in TDLib, converts it and returns result. Otherwise returns None
        """

        listener = self.app.lhub.get_listener(post.tg_channel_id)
        if not listener: return None

        tdmessage = listener.tdutil.get_post(tg_channel_id=post.tg_channel_id, tg_msg_id=post.tg_msg_id)
        if not tdmessage:
            self.app.dba.set_content_downloaded(post, False)
            #print(f'post {post} not found, setting content_downloaded=0')
            return None

        chat_title = listener.tdutil.get_chat_title(tdmessage)
        messagelink = listener.tdutil.get_message_link(tdmessage)
        res = self.app.conv.convert(
            user_id = user.id,
            post_id = post.id,
            tg_user_id = user.tg_user_id,
            message = tdmessage,
            sourcename = chat_title,
            source = messagelink
        )
        return res

        
    def got_admin_forward(self,message):
        """ this is called when bot gets forwarded message from listener """
        
        # find original post in OlegDB and flag content_downloaded

        # keep in mind that if post was originnaly forwarded to the channel Oleg is subscribed to,
        # and then forwarded by Listener to Bot, Oleg will not find it in DB, 
        # because message's 'forward_info' points to the original post (which is not in Oleg's DB most likely)

        tg_msg_id = message.get('forward_info',{}).get('origin',{}).get('message_id',0)
        tg_channel_id = message.get('forward_info',{}).get('origin',{}).get('chat_id',0)
        
        posts = self.app.dba.get_posts(tg_msg_id = tg_msg_id, tg_channel_id = tg_channel_id)
        
        if posts:
            self.app.dba.set_content_downloaded(posts[0])

    def callback_query_handler(self,update):
        """ handles callbacks from inline keyboards """
        
        payload = update.get('payload',{}).get('data','')  # get base64 string

        # decode payload from base64 encoded string
        # this will return string only if it was properly hashed:
        payload = self.app.hash.hash_b64decode(payload)
        
        if payload:
            route = payload.split(' ')[0]
            if route == 'VOTE':
                self.got_callback_reaction(payload, query_id = update.get('id',0))
            elif route == 'COMMAND':
                self.got_callback_command(payload, query_id = update.get('id',0))
        else:
            self.tdutil.answer_callback_query(query_id = update.get('id',0), text = 'No-no-no...')

    def edit_inline_keyboard(self, user_id, post_id, reaction_id):
        """ edits inline keyboard with sign of chosen reaction """
        
        # TODO
        pass

    def got_callback_reaction(self,payload,query_id):
        
        user_id, post_id, reaction_id = payload.split(' ')[1:4] # decompose string
        user_id = int(user_id)
        post_id = int(post_id)
        reaction_id = int(reaction_id)
        self.app.dba.update_reaction(user_id, post_id, reaction_id)
        #self.edit_inline_keyboard(user_id,post_id, reaction_id)
        self.tdutil.answer_callback_query(query_id = query_id, text = self.reactions[reaction_id].text)
        self._users_send_new([self.app.dba.get_user(user_id=user_id)])
        self.app.nn.got_new_reaction()

    def got_callback_command(self, payload, query_id):

        payload = payload.split(' ')
        command = payload[2]
        user_id = payload[1]

        user = self.app.dba.get_user(user_id = user_id)

        if command == 'send_new':
            self._users_send_new([user])
            self.tdutil.answer_callback_query(query_id = query_id, text = 'Here you go')

    def scheduled_mailing(self):
        """ This is called when scheduled time of mass mailing come """

        users = self.app.dba.get_users()
        self._users_send_new(users)







class Listener():
    """ rules listener's TDLib account """

    def __init__(
        self,
        api_id,
        api_hash,
        phone,
        database_encryption_key,
        files_directory,
        message_handler
    ):

        self.tdutil = TDLibUtils(
            api_id = api_id,
            api_hash = api_hash,
            phone = phone,
            database_encryption_key = database_encryption_key,
            files_directory = files_directory
        )

        # set message handler
        self.tdutil.add_message_handler(Filters.all, message_handler)

    def start(self):
        
        self.tdutil.start()
        self.user_id = self.tdutil.get_me()

    def stop(self):
        self.tdutil.stop()




        
class ListenerHub():
    """ routes queries to their Listeners """
    
    def __init__(
        self,
        api_id,
        api_hash,
        database_encryption_key,
        files_directory,
        max_listener_channels
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.database_encryption_key = database_encryption_key
        self.files_directory = files_directory
        self.max_listener_channels = max_listener_channels
        
    def start(self):

        # read listeners from DB
        listeners = self.app.dba.get_listeners()

        # instantiate Listeners, self.ls is dict where keys are Listener.id
        self.ls = {}
        for l in listeners:
            self.ls[l] = Listener(
                self.api_id,
                self.api_hash,
                listeners[l].phone,
                self.database_encryption_key,
                os.path.join(self.files_directory, str(listeners[l].id)),
                self.handle_new_message
            )
            self.ls[l].id = l
            self.ls[l].start()

    def stop(self):
        for l in self.ls:
            self.ls[l].stop()
        
    
    def handle_new_message(self, message):
        """
        called as new message comes to one of the Listeners
        checks if message is sent from a chat in a filter, and type supported
        """
        
        # get message sender id    
        sender_type = message.get('sender',{}).get('@type','')

        #print('NEW MESSAGE\nhandler ok')
        if sender_type == 'messageSenderChat':
            
            #print('sender_type ok')
            #print('Chat: ', self.tdutil.get_chat_title(message))
            #print('Text:', message.get('content',{}).get('text',{}).get('text','')[:20])
            
            sender_id = message.get('sender',{}).get('chat_id',0)

            # check if we listen to sender_id
            # and @type of content is supported and add post to OlegDB
            if (self._listening(sender_id)) and (self._type_supported(message)):
                #print('chat listening and type supported ok')

                # write new post to OlegDB
                post = self._prepare_oleg_post(message)
                #DEBUG
                self.app.debug.lhub_post = post
                post = self.app.dba.add_post(post)
                self.app.debug.lhub_post1 = post

                # add new post to model - will be added on model reinit (with cache renew)
                #self.app.nn.handle_new_obj(where='post', obj = post)

                listener = self.get_listener(sender_id)
                # forward new post to bot, so bot will have access to media
                res = listener.tdutil.forward_post(
                    from_msg_id = post.tg_msg_id,
                    from_channel_id = post.tg_channel_id,
                    to_user_id = self.app.bot.user_id
                )
                if res.error:
                    self.app.logger.error(res.error_info)

    def _type_supported(self, message):
        """ takes TDLib.message and checks if type of content is supported by OlegAI """
        
        if ((message.get('@type','') == 'message') and
            (message.get('content',{}).get('@type','') in self.app.conv.supported)):
            return True
        else:
            return False

    def _listening(self, sender_id):
        """ takes chat_id and checks if we listen to it """
        channel = self.app.dba.get_channels(tg_channel_id=sender_id)
        listening = False
        if channel:
            listening = channel[0].listening
        return listening

        
    def _prepare_oleg_post(self, message):
        """ takes post as TDLib.message and returns olegtypes.Post """
            
        # prepare post info
        tg_msg_id = message.get('id',0)
        tg_channel_id = message.get('chat_id',0)
        tg_timestamp = message.get('date',0)

        # create Post
        post = Post(
            tg_msg_id = tg_msg_id,
            tg_channel_id = tg_channel_id,
            tg_timestamp = tg_timestamp
        )
        return post

    def join_chat_mute(self, chat_id=None, link=None):
        """
        Selects listener to join.
        Joins chat by chat_id or link and mutes it.

        Returns:
        joined (bool): if joined successful
        result (dict): tg.result to be able to handle tg.result.error 
        channel (Channel): if joined, returns this channel as Channel object
        """

        listener, qty = self.get_min_listener(with_qty=True)

        if qty >= self.max_listener_channels:
            self.app.logger.warning('All listeners are full!')
            res = SimpleNamespace(error=True, error_info={'code': 1001, 'message': 'All listeners are full'})
            return False, res, None

        if chat_id is not None:
            result = listener.tdutil.join_chat(chat_id)
            joined = result.ok_received
        elif link is not None:
            joined = False
            result = listener.tdutil.join_chat_by_link(link)
            if not result.error:
                chat = result.update
                chat_id = chat.get('id',0)
                joined = True

        channel = None
        if joined:
            name = listener.tdutil.get_chat_title(chat_id = chat_id)
            channel = self.app.dba.add_channel(Channel(
                tg_channel_id = chat_id,
                listening = True,
                name = name,
                listener_id = listener.id
            ))
            self.app.nn.handle_new_obj('channel', channel)
            time.sleep(1)
            listener.tdutil.mute_chat(chat_id)
        
        return joined, result, channel

    def get_min_listener(self, with_qty=False):
        """
        Returns listener with minimum channels.
        If with_qty=True, returns tuple (listener, qty of his channels)
        """

        lv = self.app.dba.get_listeners_volume()
        listener = min(lv, key=lv.get)
        if with_qty:
            return self.ls[listener], lv[listener]
        else:
            return self.ls[listener]

    def get_listener(self, tg_channel_id):
        """ Returns listener for tg_channel_id """

        channel = self.app.dba.get_channels(tg_channel_id = tg_channel_id)
        if channel:
            channel = channel[0]
        else:
            return None

        lid = channel.listener_id
        return self.ls[lid]