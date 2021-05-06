from types import SimpleNamespace

class BaseDBObject(SimpleNamespace):
    def __init__(self,**kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
        

class User(BaseDBObject):
    """ user """
    table_name = 'users'
    cols = [
        'id',
        'tg_user_id',
        'timestamp',
        'username',
    ]


class Post(BaseDBObject):
    """ OlegDB post """
    table_name = 'posts'
    cols = [
        'id',
        'tg_msg_id',
        'tg_channel_id',
        'tg_timestamp',
        'content_downloaded',
        'timestamp'
    ]
        
class Reaction(BaseDBObject):
    """ reaction  """
    table_name = 'reactions'
    cols = [
        'id',
        'emoji',
        'text',
    ]

class UserReaction(BaseDBObject):
    """ user's reaction to given post """
    table_name = 'user_reactions'
    cols = [
        'id',
        'user_id',
        'internal_post_id',
        'reaction_id',
        'learned',
        'timestamp',
    ]

class Repost(BaseDBObject):
    """ repost"""

    table_name = 'reposts'
    cols = [
        'id',
        'internal_post_id',
        'user_id',
        'timestamp'
    ]

class Channel(BaseDBObject):
    """ channel """
    table_name = 'channels'
    cols = [
        'id',
        'tg_channel_id',
        'listening',
        'name',
        'timestamp',
    ]
  
class ChannelPool(BaseDBObject):
    """ channel from channel_pool """
    table_name = 'channel_pool'
    cols = [
        'id',
        'tgstat_id',
        'internal_channel_id',
        'tg_id',
        'link',
        'username',
        'title',
        'about',
        'participants_count',
        'tgstat_restrictions',
        'chat_id',
        'chat_id_failed',
        'joined',
        'error_code']

class MessageHandler(SimpleNamespace):
    """ TDLibUtils messages handler """
    def __init__(self,
                 filter,
                 handler
                ):
        self.filter = filter
        self.handler = handler

class CommandHandler(SimpleNamespace):
    """ TDLibUtils commands handler """
    def __init__(self,command,handler,fltr):
        self.command = command
        self.handler = handler
        self.filter = fltr
