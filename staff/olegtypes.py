from types import SimpleNamespace

class BaseDBObject(SimpleNamespace):
    def __init__(self,**kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
        

class User(BaseDBObject):
    """ user """
    pass


class Post(BaseDBObject):
    """ OlegDB post """
    pass
        
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

class Reaction(BaseDBObject):
    """ reaction  """
    pass

class UserReaction(BaseDBObject):
    """ user's reaction to given post """
    pass

class Repost(BaseDBObject):
    """ repost"""

    table_name = 'reposts'
    cols = [
        'id',
        'internal_post_id',
        'user_id'
    ]

class Channel(BaseDBObject):
    """ channel """
    pass
  
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