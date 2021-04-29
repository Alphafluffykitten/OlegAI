from types import SimpleNamespace

class BaseDBObject(SimpleNamespace):
    def __init__(self,**kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
        

class User(BaseDBObject):
    """ object representing user """
    pass


class Post(BaseDBObject):
    """ object representing OlegDB post """
    pass
        
class MessageHandler(SimpleNamespace):
    """ object representing TDLibUtils messages handler """
    def __init__(self,
                 filter,
                 handler
                ):
        self.filter = filter
        self.handler = handler

class CommandHandler(SimpleNamespace):
    """ object representing TDLibUtils commands handler """
    def __init__(self,command,handler,fltr):
        self.command = command
        self.handler = handler
        self.filter = fltr

class Reaction(BaseDBObject):
    """ object representing reaction  """
    pass

class UserReaction(BaseDBObject):
    """object representing user's reaction to given post """
    pass

class Repost(BaseDBObject):
    """object representing repost"""
    pass

class Channel(BaseDBObject):
    """ object representing Channel """
    pass
  
class ChannelPool(BaseDBObject):
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