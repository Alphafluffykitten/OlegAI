from staff.filters import Filters
from staff.olegtypes import *
from tdlib_client.client import Telegram
import json
import ast
import base64
import hashlib
import time, datetime


# Error codes:
# 1001 - all listeners are full


class TDLibUtils():
    """ TDLib helpers """
    
    def __init__(self,
                 api_id,
                 api_hash,
                 database_encryption_key,
                 phone=None,
                 bot_token=None,
                 files_directory=None,
                ):
        
        self._command_handlers = []
        self._message_handlers = []
        self.greeting = None
        
        self.tg = Telegram(api_id = api_id,
                           api_hash = api_hash,
                           database_encryption_key = database_encryption_key,
                           bot_token = bot_token,
                           phone = phone,
                           files_directory = files_directory)
        
        
    def start(self):
        self.tg.login()
        self.user_id = self.get_me()

        # listen to any update.message and pass it to update_handler
        self.tg.add_message_handler(self.update_handler)

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

        sender = self.get_sender(m)
        if sender == self.user_id: return
        
        # code that'll execute upon receiving any message
        if self.greeting and m:
            self.greeting(m)

        # parse bot commands
        command, params = self._parse_commands(m)
        
        if command: 
            self._apply_command_filter(m, command, params)
        else:
            self._apply_msg_filter(m)

    def _parse_commands(self, message):
        """ parses bot commands from TDLib.message, returns a list without '/' """
        command, params = (None,None)

        if message.get('content',{}).get('@type','') == 'messageText':
            text = message.get('content',{}).get('text',{}).get('text','')
            entities = message.get('content',{}).get('text',{}).get('entities',[])
            
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
            #print(result.error_info)
            pass

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
        """ Sends TDLib.sendMessage """
        result = self._send_data('sendMessage', message)
        return result

    def send_message_album(self, message):
        result = self._send_data('sendMessageAlbum', message)
        return result

    def send_text(self, tg_user_id, text):
        """ Sends text message to tg_user_id """
        return self.tg.send_message(tg_user_id, text)
        
    
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

    def get_sender(self,message):
        """ Returns sender's tg_user_id """
        tg_user_id = None
        if message.get('sender',{}).get('@type','') == 'messageSenderUser':
            tg_user_id = message.get('sender',{}).get('user_id', 0)
        return tg_user_id

    def get_username(self, tg_user_id):
        """ returns username of Telegram user by his TG user_id """

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

    def download_file(self, file_id):
        data = {
            'file_id': file_id,
            'priority': 32,
            'offset': 0,
            'limit': 0,
            'synchronous': True
        }
        return self._send_data('downloadFile', data)







class Converter():
    """ 
    Takes TDLib.message and makes TDLib.sendMessage
    Provides some helper methods to append stuff to the message
    """

    # TODO: take care of albums
    
    def __init__(self):

        self.tconv = TypeConverters()

        # read supported methods
        self.supported = [f for f in dir(TypeConverters) if not f.startswith('_')]
    
    def convert(self,message):
        """
        Converts message into TDLib.sendMessage
        To get message ready to send to user, use append_recepient_user

        Args:
        message (dict): TDLib.message with content to be converted
        """
        
        content_type = message.get('content',{}).get('@type','')
        if content_type in self.supported:
            result = getattr(self.tconv, content_type)(message)
        else:
            return False
        
        return result

    def make_send_message(self, input_message_content):
        m = {
            '@type': 'sendMessage',
            'input_message_content': input_message_content
        }
        return m
    
    def make_send_message_album(self,input_message_contents):
        m = {
            '@type': 'sendMessageAlbum',
            'input_message_contents': input_message_contents
        }
        return m

    def append_recepient_user(self, message, user):
        m = message.copy()
        m['chat_id'] = user.tg_user_id
        return m
    
    def append_source_info_as_text_url(self, message,sourcename,source):
        """ appends source link at the end of TDLib.sendMessage text or caption"""
        
        m = message.copy()
        
        mcontent = m.get('input_message_content',{})
        
        if 'text' in mcontent:
            textcaption = 'text'
        elif 'caption' in mcontent:
            textcaption = 'caption'
        else:
            raise Exception('[ Converter._append_source_info ] No text or caption content')
            
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

    def append_source_info_as_button(self, message,sourcename,source):

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

    def append_inline_keyboard(self, message, user_id, post_id):
        """ appends reply buttons to the TDLib.sendMessage """
        
        m = message.copy()
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

    def append_moar_button(self,message,user_id):
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
    

class TypeConverters():
    """ 
    Type converters for Converter
    Takes TDLib.message and returns corresponding to its type TDLib.inputMessageContent
    """

    def messageText(self,message):
       
        post = {
            '@type': 'inputMessageText',
            'text': message.get('content',{}).get('text',{}),
            'disable_web_page_preview': (not bool(message.get('content',{}).get('web_page',{})))
        }
        
        return post
    
    def messagePhoto(self,message):
        
        maxsize = self._get_photo_max_size(message)
            
        post = {
            '@type': 'inputMessagePhoto',
            'photo': {
                '@type': 'inputFileRemote',
                'id': maxsize.get('photo',{}).get('remote',{}).get('id',0)
            },
            'width': maxsize.get('width',0),
            'height': maxsize.get('height',0),
            'caption': message.get('content',{}).get('caption',{})
        }
    
        
        return post

    def _get_photo_max_size(self, message):
        # search for max size
        sizes = message.get('content',{}).get('photo',{}).get('sizes',[])
        w = []
        for s in sizes:
            w.append(s.get('width',0))
        maxsizeindex = w.index(max(w))
        maxsize = sizes[maxsizeindex]

        return maxsize


    def messageVideo(self,message):

        video = message.get('content',{}).get('video',{})
        post = {
            '@type': 'inputMessageVideo',
            'video': {
                '@type': 'inputFileRemote',
                'id': video.get('video',{}).get('remote',{}).get('id',0),
            },
            'duration': video.get('duration',0),
            'width': video.get('width',0),
            'height': video.get('height',0),
            'caption': message.get('content',{}).get('caption',{})
        }
        
        return post

    def messageAnimation(self,message):
        
        animation = message.get('content',{}).get('animation',{})
        post = {
            '@type': 'inputMessageAnimation',
            'animation': {
                '@type': 'inputFileRemote',
                'id': animation.get('animation',{}).get('remote',{}).get('id',0),
            },
            'duration': animation.get('duration',0),
            'width': animation.get('width',0),
            'height': animation.get('height',0),
            'caption': message.get('content',{}).get('caption',{}),
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
        data = data.encode('ascii')   #string with truncated hash goes to payload
        data = base64.b64encode(data) #encode bytes to base64 bytes
        data = data.decode('ascii')   #decode base64 bytes to ascii representation

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

class PostsCache():
    """ Provides smart cache for .posts and set of post ids which should go into model under .long """

    def __init__(self, dba, cache_days=10):
        self.dba = dba
        self.cache_days = cache_days
    
    def renew(self):
        self.long = self.dba.get_user_reactions_post_ids()

        days_ago = (datetime.datetime.now() - datetime.timedelta(days=self.cache_days)).timestamp()
        self.posts = self.dba.get_posts(
            have_content = True,
            tg_timestamp_range = (days_ago, time.time())
        )
        short = set()
        for p in self.posts:
            short.add(p.id)

        self.long = self.long | short
