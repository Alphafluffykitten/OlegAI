from tdlib_client.client import Telegram
from staff.olegtypes import *
from staff.dba import OlegDBAdapter
from staff.filters import Filters
from staff.nn import OlegNN
from staff.crawler import Joiner, CrawlerDB
import json
import ast
import base64
import hashlib
import torch
import time, datetime
import os
from apscheduler.schedulers.background import BackgroundScheduler


def dumps(o):
    return json.dumps(o,indent=4,ensure_ascii=False)

#TODO: smear the whole app with logs
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
                 admin_phone,
                 admin_db_encryption_key,
                 bot_db_encryption_key,
                 admin_files_directory,
                 bot_files_directory,
                 salt,
                 logs_dir,
                 admin_user_id,
                 nn_lpv_len,
                 nn_new_reactions_threshold,
                 nn_learning_timeout,
                 nn_full_learn_threshold,
                ):

        self.logs_dir = logs_dir
        self._setup_logs_dir()
        
        self.dba = OlegDBAdapter(db_name,db_user,db_password,db_host)
        self.conv = Converters()
        self.hash = OlegHashing(salt = salt, trunc = 7)
        self.nn = OlegNN(
            nn_lpv_len,
            nn_new_reactions_threshold,
            nn_learning_timeout,
            nn_full_learn_threshold,
        )

        self.admin = Admin(
            api_id = api_id,
            api_hash = api_hash,
            phone = admin_phone,
            database_encryption_key = admin_db_encryption_key,
            files_directory = admin_files_directory,)
        
        self.bot = Bot(
            api_id = api_id,
            api_hash = api_hash,
            bot_token = bot_token,
            database_encryption_key = bot_db_encryption_key,
            files_directory = bot_files_directory,
            admin_user_id = admin_user_id,)

        self.crawler_db = CrawlerDB(db_name,db_user,db_password,db_host)
        self.joiner = Joiner(self.crawler_db, self.admin, logs_dir = logs_dir)
        
        self.scheduler = BackgroundScheduler(timezone = 'Europe/Moscow')
        self.scheduled = self.scheduler.add_job(
            func = self.bot.scheduled_mailing,
            trigger = 'cron',
            hour = '10,12,14,16,19',
            minute = '20',
            coalesce = True,
        )

        # attach OlegApp to sub-objects so they have access to other inhabitants of app
        self.bot.app = self
        self.admin.app = self
        self.nn.app = self
        self.conv.app = self


    def start(self):
        """ should be called before anything """
        
        self.dba.start()
        self.nn.start()
        self.admin.start()
        self.bot.start()
        self.scheduler.start()
        self.crawler_db.start()

        
        
    def stop(self):
        """ graceful shutdown """
        
        self.joiner.stop()
        self.crawler_db.stop()
        self.scheduler.shutdown(wait=False)
        self.bot.stop()
        self.admin.stop()
        self.dba.stop()

    def _setup_logs_dir(self):
        if not os.path.isdir(self.logs_dir):
            os.mkdir(self.logs_dir)

        
        
class Admin():
    """ functions to use with admin TDLib account """
    
    def __init__(self,
                 api_id,
                 api_hash,
                 phone,
                 database_encryption_key,
                 files_directory,
                ):
        
        self.tdutil = TDLibUtils(api_id = api_id,
                                 api_hash = api_hash,
                                 phone = phone,
                                 database_encryption_key = database_encryption_key,
                                 files_directory = files_directory
                                )
        
        
        
    def start(self):
        # set admin message handler
        self.tdutil.add_message_handler(Filters.all, self.new_message_handler)
        
        self.tdutil.start()

        self.user_id = self.tdutil.get_me()
        

    def stop(self):
        self.tdutil.stop()
        
    
    def new_message_handler(self, message):
        """
        called as new message comes to Admin
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

                # check for emb from previous post from this channel
                prev = self.app.dba.get_posts(tg_channel_id = sender_id, have_content=True, limit = 100)
                emb = None
                if prev:
                    emb, bias = self.app.nn.mean_of_posts(prev)
                    #print('got existing emb: ', emb[:5], bias,sep='\n')
                if not isinstance(emb,torch.Tensor):
                    emb = self.app.nn.get_init_emb()
                    bias = torch.tensor([0])

                # write new post to OlegDB
                post = self._prepare_oleg_post(message)
                post = self.app.dba.add_post(post)

                # update embeddings
                self.app.nn.add_emb(where = 'post', post = post, emb = emb, bias = bias)
                # forward new post to bot, so bot will have access to media
                res = self.tdutil.forward_post(
                    from_msg_id = post.tg_msg_id,
                    from_channel_id = post.tg_channel_id,
                    to_user_id = self.app.bot.user_id
                )
                if res.error:
                    print(res.error_info)

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
        post = Post( tg_msg_id = tg_msg_id,
                     tg_channel_id = tg_channel_id,
                     tg_timestamp = tg_timestamp
                   )
        return post

    def join_chat_mute(self, chat_id=None, link=None):
        """
        Joins chat by chat_id or link and mutes it.

        Returns:
        joined (bool): if joined successful
        result (dict): tg.result to be able to handle tg.result.error 
        channel (Channel): if joined, returns this channel as Channel object
        """

        if chat_id is not None:
            result = self.tdutil.join_chat(chat_id)
            joined = result.ok_received
        elif link is not None:
            joined = False
            result = self.tdutil.join_chat_by_link(link)
            if not result.error:
                chat = result.update
                chat_id = chat.get('id',0)
                joined = True

        channel = None
        if joined:
            name = self.tdutil.get_chat_title(chat_id = chat_id)
            channel = self.app.dba.add_channel(Channel(tg_channel_id = chat_id, listening = True, name = name))
            time.sleep(1)
            self.tdutil.mute_chat(chat_id)
        
        return joined, result, channel
        







class TDLibUtils():
    """ TDLib helpers """
    
    def __init__(self,
                 api_id,
                 api_hash,
                 database_encryption_key,
                 phone=None,
                 bot_token=None,
                 files_directory=None
                ):
        
        self._command_handlers = []
        self._message_handlers = []
        self.greeting = None
        
        self.tg = Telegram(api_id = api_id,
                           api_hash = api_hash,
                           database_encryption_key = database_encryption_key,
                           bot_token = bot_token,
                           phone = phone,
                           files_directory = files_directory
                          )
        
        # listen to any update.message and pass it to update_handler
        self.tg.add_message_handler(self.update_handler)
        
    def start(self):
        self.tg.login()
        
    def stop(self):
        self.tg.stop()
        
    def _send_data(self,method,data):
        result = self.tg.call_method(method, data)
        result.wait()

        return result
        
    def add_command_handler(self, command, handler, fltr=Filters.all):
        ''' adds command handler '''
        self._command_handlers.append(CommandHandler(command,handler,fltr))

    def add_message_handler(self,filter,handler):
        ''' adds message handler '''
        self._message_handlers.append(MessageHandler(filter,handler))
        
    def update_handler(self, update):
        """ routes different kinds of message updates to its handlers """
        
        m = update.get('message',{})
        
        # code that'll execute upon receiving any message
        if self.greeting and m:
            self.greeting(m)

        sender_type = m.get('sender',{}).get('@type','')

        # parse bot commands
        command, params = self._parse_commands(m)
        
        if command: 
            self._apply_command_filter(m, command, params)
        else:
            self._apply_msg_filter(m)

    def _parse_commands(self,message):
        """ parses bot commands from TDLib.message, returns a list without '/' """
        
        text = message.get('content',{}).get('text',{}).get('text','')
        entities = message.get('content',{}).get('text',{}).get('entities',[])
        
        command, params = (None,None)
        for e in entities:
            if e.get('type',{}).get('@type','') == 'textEntityTypeBotCommand':
                offset = e.get('offset',0)
                start = offset + 1
                end = offset + e.get('length',0)
                command = text[start:end]
                # parse command params
                params = (text[end:]).split(' ')
                params = list(filter(bool,params))
                break
        return command, params

    def _apply_msg_filter(self,message):
        """ reads all handlers from self._message_handlers and takes first whose filter returns True """
        
        for h in self._message_handlers:
            if h.filter(message):
                h.handler(message)
                break

    def _apply_command_filter(self, message, command, params):
        """ reads all handlers from self._command_handlers and takes first whose filter returns True """

        for h in self._command_handlers:
            if h.command == command:
                if h.filter(message):
                    h.handler(message,params)


    def get_post(self, tg_channel_id, tg_msg_id):
        """ returns post from tdlib """
        
        data = {'chat_id' : tg_channel_id,
                'message_id': tg_msg_id
               }
        result = self._send_data('getMessage', data)
        if result.error:
            print(result.error_info)

        if result.update:
            return result.update
    
    def get_chat_title(self, message=None, chat_id = None):
        """ returns chat title from TDLib based on TDLib.message or chat_id """
        
        if message is not None:
            data = {'chat_id': message.get('sender',{}).get('chat_id',None)}
        elif chat_id is not None:
            data = {'chat_id': chat_id}
        else:
            raise Exception('[ TDLibUtils.get_chat_title ]: Have to provide either message or chat_id')
        result = self._send_data('getChat',data)
        if result.update: 
            return result.update.get('title','')
        
    def get_message_link(self,message):
        """ returns link to message based on TDLib.message """
        
        data = {'chat_id': message.get('sender',{}).get('chat_id',None),
                'message_id': message.get('id',None)
               }
        result = self._send_data('getMessageLink',data)
        if result.update:
            return result.update.get('link','')
        
    
    def send_message(self, message):
        """ takes TDLib.sendMessage and sends it """
        #print(dumps(message))
        result = self._send_data('sendMessage', message)
        return result
        
    
    def forward_post(self, from_msg_id, from_channel_id, to_user_id):
        ''' forwards post '''
        
        data = {
            'chat_id': to_user_id,
            'from_chat_id': from_channel_id,
            'message_ids': [from_msg_id]
        }
        
        return self._send_data('forwardMessages',data)
        
        
    def get_me(self):
        ''' returns user_id of this Telegram user '''
        
        data = {}
        result = self._send_data('getMe',data)
        if result.update:
            update = result.update
            user = update.get('id',0)
            return user

    def get_username(self, tg_user_id):
        """ returns username of Telegram user by his TG id """

        data = {
            'user_id': tg_user_id
        }
        result = self._send_data('getUser', data)
        if result.update:
            return result.update.get('username','')

    def answer_callback_query(self, query_id, text):
        data = {
            'callback_query_id': query_id,
            'text': text,
            'show_alert': False,
        }
        
        result = self._send_data('answerCallbackQuery',data)
        if result.update: 
            return result.update 

    def join_chat(self, chat_id):
        data = {
            'chat_id': chat_id
        }

        result = self._send_data('joinChat',data)

        return result

    def join_chat_by_link(self, link):
        data = {
            'invite_link': link
        }
        return self._send_data('joinChatByInviteLink',data)

    def mute_chat(self, chat_id):
        data = {
            'chat_id': chat_id,
            'notification_settings': {
                'mute_for': 1000000
            }
        }
        self._send_data('setChatNotificationSettings',data)

    def search_public_chat(self, tg_username):
        """ takes Telegram username as arg and returns raw tg.result to be able to handle result.error  """
        data = {
            'username': tg_username
        }
        return self._send_data('searchPublicChat', data)


    def check_chat_invite_link(self, tg_joinchat):
        """ takes Telegram joinchat link as arg and returns raw tg.result to be able to handle result.error """
        data = {
            'invite_link': tg_joinchat
        }
        return self._send_data('checkChatInviteLink',data)








                
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
        admin_user_id,):
        
        self.tdutil = TDLibUtils(
            api_id = api_id,
            api_hash = api_hash,
            database_encryption_key = database_encryption_key,
            bot_token = bot_token,
            files_directory = files_directory)

        self.admin_user_id = admin_user_id
        
    def start(self):
        
        # this executes upon receiving any update from TDLib
        self.tdutil.greeting = self.user_greeting

        # set bot handlers
        #self.tdutil.add_command_handler('start',self.start_handler)
        self.tdutil.add_command_handler('send_new',self.send_new_handler)
        self.tdutil.add_command_handler('send_channel',self.send_channels_last)
        self.tdutil.add_command_handler('start_joiner', self.start_joiner_handler, Filters.user(self.admin_user_id))
        self.tdutil.add_command_handler('stop_joiner', self.stop_joiner_handler, Filters.user(self.admin_user_id))
        self.tdutil.add_message_handler(Filters.forwarded & Filters.user(self.app.admin.user_id), self.got_admin_forward)
        self.tdutil.tg.add_update_handler('updateNewCallbackQuery', self.callback_query_handler)
        
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
                emb = self.app.nn.get_init_emb()
                user = self.app.dba.register_user(tg_user_id, username)
                if user:
                    # add user embedding and bias to nn
                    self.app.nn.add_emb(where = 'user', user = user, emb = emb, bias = torch.tensor([0]))
                    self.user_bootstrap(user)
                    self._user_send_new(user)
                    
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
        time.sleep(2)
        
    def start_handler(self, message, params):
        """ on /start """

        tg_user_id = message.get('chat_id',0)
        
        #self.tdutil.tg.send_message(chat_id=tg_user_id, text='Hi!')
        
    
    def send_new_handler(self, message, params):
        """ sends new post when user asks with /send_new """
        
        tg_user_id = message.get('chat_id',0)
        user = self.app.dba.get_user(tg_user_id=tg_user_id)
        self._user_send_new(user)

    def send_channels_last(self, message, params):
        """ sends last post from given channel to user from which message was received """

        tg_user_id = message.get('chat_id',0)
        user = self.app.dba.get_user(tg_user_id = tg_user_id)
        post = self.app.dba.get_posts(tg_channel_id = params[0],have_content=True,limit=1)
        if post:
            post = post[0]

            to_send = self._prepare_message(user,post)

            res = self.tdutil.send_message(to_send)
            if res.error:
                print(res.error_info)
        else:
            res = self.tdutil.tg.send_message(chat_id=user.tg_user_id, text=f'No posts found from channel {params[0]}')

    def start_joiner_handler(self,message, params):
        self.app.joiner.start()

    def stop_joiner_handler(self, message, params):
        self.app.joiner.stop()

    def _user_send_new(self, user:User):
        """ selects post from pool of unseen and sends it to user_id """

        ten_days_ago = (datetime.datetime.now() - datetime.timedelta(days=10)).timestamp()
        # if post's not found in TDLib, delete it, try 10 times, if fails, just pass
        for i in range(10):
            # if user is fresh, feed him posts with highest bias (sweetest shit we have)
            if self.app.dba.count_reposts(user) < 30:
                max_bias_posts = self.app.dba.get_posts(ids=self.app.nn.get_max_bias())
                # get max biased post that havent been reposted to this user
                for p in max_bias_posts:
                    if (
                        (not self.app.dba.get_reposts(post_id = p.id, user_id=user.id))
                        and (p.content_downloaded)
                    ):
                        break
                filtered = p
            else:
                # select posts within last 10 days
                posts = self.app.dba.get_posts(
                    have_content = True,
                    except_user = user.id,
                    tg_timestamp_range = (ten_days_ago, time.time())
                )
                if not posts:
                    self.tdutil.tg.send_message(chat_id=user.tg_user_id, text='No new messages :(')
                    return

                filtered = self.app.nn.closest(posts,user)

            to_send = self._prepare_message(user, filtered)
            if to_send: break

        if not to_send: return

        res = self.tdutil.send_message(to_send)
        if res.error:
            print(res.error_info)

        # write reposted info to OlegDB.reposts
        self.app.dba.add_repost(filtered,user)

    def _prepare_message(self,user,post):
        """
        prepares tdmessage for sending (name, converters etc)

        Returns: if message was found in TDLib, converts it and returns result. Otherwise returns None
        """

        tdmessage = self.app.admin.tdutil.get_post(tg_channel_id=post.tg_channel_id, tg_msg_id=post.tg_msg_id)
        if not tdmessage:
            self.app.dba.set_content_downloaded(post, False)
            #print(f'post {post} not found, setting content_downloaded=0')
            return None

        chat_title = self.app.admin.tdutil.get_chat_title(tdmessage)
        messagelink = self.app.admin.tdutil.get_message_link(tdmessage)
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
        """ this is called when bot gets forwarded message from admin user """
        
        # find original post in OlegDB and flag content_downloaded

        # keep in mind that if post was originnaly forwarded to the channel Oleg is subscribed to,
        # and then forwarded by Admin to Bot, Oleg will not find it in DB, 
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
        self._user_send_new(self.app.dba.get_user(user_id=user_id))
        self.app.nn.got_new_reaction()

    def got_callback_command(self, payload, query_id):

        payload = payload.split(' ')
        command = payload[2]
        user_id = payload[1]

        user = self.app.dba.get_user(user_id = user_id)

        if command == 'send_new':
            self._user_send_new(user)
            self.tdutil.answer_callback_query(query_id = query_id, text = 'Here you go')

    def scheduled_mailing(self):
        """ this is called when scheduled time of mass mailing of new posts come """

        users = self.app.dba.get_users()
        for u in users:
            self._user_send_new(u)
        










class Converters():
    """ 
        takes TDLib.message and makes TDLib.sendMessage with source link appended

        Notes:
        - cannot pass attached message.content.web_page to the inputMessageText since
        there is no such attribute when you send a message
    """

    # TODO: take care of albums
    
    def __init__(self):
        # read supported methods from this class
        self.supported = [f for f in dir(Converters) if not f.startswith('_')]
    
    def convert(self, user_id, post_id, tg_user_id, message, sourcename, source):
        """ router method """
        
        content_type = message.get('content',{}).get('@type','')
        if content_type in self.supported:
            result = getattr(self, content_type)(message)
        else:
            raise Exception(f'[ Converters.convert ]: Content @type {content_type} not supported')
        
        result['chat_id'] = tg_user_id
        result = self._append_inline_keyboard(result,user_id,post_id)
        result = self._append_source_info_as_text_url(result,sourcename,source)
        #result = self._append_moar_button(result,user_id)
        
        return result
    
    def _append_source_info_as_text_url(self, message,sourcename,source):
        """ appends source link at the end of TDLib.sendMessage text or caption"""
        
        m = message.copy()
        
        mcontent = m.get('input_message_content',{})
        
        if 'text' in mcontent:
            textcaption = 'text'
        elif 'caption' in mcontent:
            textcaption = 'caption'
        else:
            raise Exception('[ Converters._append_source_info ] No text or caption content')
            
        messagetext = m.get('input_message_content',{}).get(textcaption,{}).get('text','')
        
        offset = len(ast.literal_eval(json.dumps(messagetext)))
        txt = f'\n\nSource: {sourcename}'
        m['input_message_content'][textcaption]['text'] += txt
        
        entity = {
            '@type': 'textEntity',
            'offset': offset,
            'length': len(ast.literal_eval(json.dumps(txt))),
            'type': {
                '@type': 'textEntityTypeTextUrl',
                'url': source
            }
        }
        
        m['input_message_content'][textcaption]['entities'].append(entity)
        
        return m

    def _append_source_info_as_button(self, message,sourcename,source):

        m = message.copy()

        button = {
            'text': f'via: {sourcename}',
            'type': {
                '@type': 'inlineKeyboardButtonTypeUrl',
                'url': source
            }
        }

        if not ('reply_markup' in m):
            m['reply_markup'] = {}
            m['reply_markup']['rows'] = []
        m['reply_markup']['rows'].append([button])

        return m

    def _append_inline_keyboard(self, message, user_id, post_id):
        """ appends reply buttons to the TDLib.sendMessage """
        
        m = message
        reactions = self.app.bot.reactions

        buttons = []
        for key in reactions:
            r = reactions[key]
            data = self.app.hash.hash_b64encode(f'VOTE {user_id} {post_id} {r.id}')
            button = {
                'text' : str(r.emoji),
                'type' : {
                    '@type': 'inlineKeyboardButtonTypeCallback',
                    'data': str(data)
                }
            }
            buttons.append(button)

        m['reply_markup'] = {
            '@type': 'replyMarkupInlineKeyboard',
            'rows': [
                [buttons[0],buttons[1]],    # buttons row
            ]
        }
        return m

    def _append_moar_button(self,message,user_id):
        """ appends MOAR button to reply keyboard """

        data = self.app.bot.hash.hash_b64encode(f'COMMAND {user_id} send_new')
        button = {
            'text': 'MOAR!',
            'type': {
                '@type': 'inlineKeyboardButtonTypeCallback',
                'data': str(data)
            }
        }
         
        if not message.get('reply_markup',None):
            message['reply_markup'] = {
                '@type': 'replyMarkupInlineKeyboard',
                'rows': []
            }

        message['reply_markup']['rows'].append([button])
        return message

    def messageText(self,message):
       
        post = {
            'input_message_content': {
                '@type': 'inputMessageText',
                'text': message.get('content',{}).get('text',{}),
                'disable_web_page_preview': (not bool(message.get('content',{}).get('web_page',{})))
            }
        }
        
        return post
    
    def messagePhoto(self,message):
        
        # search for max size
        sizes = message.get('content',{}).get('photo',{}).get('sizes',[])
        w = []
        for s in sizes:
            w.append(s.get('width',0))
        maxsizeindex = w.index(max(w))
        
        maxsize = sizes[maxsizeindex]
            
        post = {
            'input_message_content': {
                '@type': 'inputMessagePhoto',
                'photo': {
                    '@type': 'inputFileRemote',
                    'id': maxsize.get('photo',{}).get('remote',{}).get('id',0)
                },
                'width': maxsize.get('width',0),
                'height': maxsize.get('height',0),
                'caption': message.get('content',{}).get('caption',{})
            }
        }
        
        return post

    def messageVideo(self,message):
        #print(dumps(message))

        video = message.get('content',{}).get('video',{})
        post = {
            'input_message_content': {
                '@type': 'inputMessageVideo',
                'video': {
                    '@type': 'inputFileRemote',
                    'id': video.get('video',{}).get('remote',{}).get('id',0),
                },
                'duration': video.get('duratoin',0),
                'width': video.get('width',0),
                'height': video.get('height',0),
                'file_name': video.get('file_name',0),
                'mime_type': video.get('mime_type',0),
                'caption': message.get('content',{}).get('caption',{})
            }
        }
        
        return post

class OlegHashing():
    """ helps hashing and testing for truth """

    def __init__(self, salt, trunc):
        self.salt = salt
        self.trunc = trunc  # leaves this many symbols of hash, if 0, leaves all

    def hash_with_salt(self, s):
        """ hashes string with salt and returns string with truncated salt appended """

        data = f'{s} {self.salt}'.encode('ascii') # encode string to raw bytes object
        hash_obj = hashlib.md5(data)              # hash it 
        if self.trunc > 0:
            hash_txt = hash_obj.hexdigest()[0:self.trunc]      # get truncated hash symbols
        else:
            hash_txt = hash_obj.hexdigest()
        return f'{s} {hash_txt}'
    
    def test_string(self, s):
        """ tests if string was hashed with proper salt """

        data = s.split(' ')

        origin = ' '.join(data[0:-1])
        if not origin:
            return False
            
        origin_hashed = self.hash_with_salt(origin)

        return origin_hashed == s

    def hash_b64encode(self,s):
        ''' hashes string, appends truncated hash, and converts to base-64 encoded string '''

        data = self.hash_with_salt(s)
        data = data.encode('ascii') #string with truncated hash goes to payload
        data = base64.b64encode(data) #encode bytes to base64 bytes
        data = data.decode('ascii') #decode base64 bytes to ascii representation

        return data

    def hash_b64decode(self,s):
        """ 
        Decodes s from base64 string to string,
        tests string for been properly hashed with salt, and returns it if yes.
        Returns empty string if not.
        """

        payload = s.encode('ascii')                        # get base64 bytes
        payload = base64.b64decode(payload)                # get raw bytes
        payload = payload.decode('ascii')                  # get string from bytes

        if self.test_string(payload):
            return payload
        else: 
            return ''
