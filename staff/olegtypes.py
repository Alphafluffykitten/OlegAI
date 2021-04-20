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
    """ object representing TDLibUtils messages filter """
    def __init__(self,
                 filter,
                 handler
                ):
        self.filter = filter
        self.handler = handler

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
  