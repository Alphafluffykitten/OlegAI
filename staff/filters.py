
class BaseFilter:
    def __and__(self,other):
        return AndFilter(self,other)
    def __or__(self,other):
        return OrFilter(self,other)
    def __invert__(self):
        return InvertFilter(self)
    
    def __call__(self,message):
        return self.filter(message)
    
class AndFilter(BaseFilter):
    def __init__(self,left,right):
        self.left = left
        self.right = right
        
    def __call__(self,message):
        left = self.left(message)
        right = self.right(message)
        return (left and right)
        
    
class OrFilter(BaseFilter):
    def __init__(self,left,right):
        self.left = left
        self.right = right
        
    def __call__(self,message):
        left = self.left(message)
        right = self.right(message)
        return (left or right)
    
class InvertFilter(BaseFilter):
    def __init__(self,arg):
        self.arg = arg
        
    def __call__(self,message):
        arg = self.arg(message)
        
        return not arg


class Filters:
    
    class _All(BaseFilter):
        """ all messages """
        
        def filter(self,message) -> bool:
            return True
    
    all = _All()
        
        
    class _Forwarded(BaseFilter):
        """ forwarded messages """
        
        def filter(self,message) -> bool:
            return bool(message.get('forward_info',{}))
        
    forwarded = _Forwarded()
    
    
    class _UserBase(BaseFilter):
        def __init__(self,chat_id):
            self.chat_id = chat_id

        def filter(self,message):
            return (str(message.get('sender',{}).get('user_id',0)) == str(self.chat_id))
    
    class _User(BaseFilter):
        """ messages from specified chat_id """

        def __call__(self,chat_id):
            return Filters._UserBase(chat_id)
            
    user = _User()
    