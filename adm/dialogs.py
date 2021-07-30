from staff.utils import Converter
from dateutil import parser
from dateutil.parser import ParserError
from zoneinfo import ZoneInfo
import json

def dumps(o):
    return json.dumps(o,indent=4,ensure_ascii=False)

class Dialog():
    """ Base class for dialogs """

    def __init__(self, user, app):
        self.user = user
        self.app = app
        self.next = None
        self.result = {}
        self.start()

    def append_reply_keyboard(self, message, buttons: list):
        """ Appends reply keyboard with simple text buttons """

        m = message.copy()

        btns = []
        for b in buttons:
            btns.append({
                '@type': 'keyboardButton',
                'text': b,
                'type': {
                    '@type': 'keyboardButtonTypeText'
                }
            })

        if btns:
            m['reply_markup'] = {
                '@type': 'replyMarkupShowKeyboard',
                'rows': [btns],
                'one_time': True,
                'resize_keyboard': True,}

        return m

    def remove_reply_keyboard(self, message):
        m = message.copy()
        m['reply_markup'] = {
            '@type': 'replyMarkupRemoveKeyboard',
        }
        return m

    def _answer(self, text, buttons=[]):
        data = {
            'chat_id': self.user.tg_user_id,
            'input_message_content': {
                '@type': 'inputMessageText',
                'text': {
                    '@type': 'formattedText',
                    'text': text,
                    'entities': []
                }
            }
        }
        if buttons:
            data = self.append_reply_keyboard(data, buttons)
        else:
            data = self.remove_reply_keyboard(data)
        
        self.app.bot.tdutil.send_message(data)

    def _get_text(self, message):
        if message.get('content',{}).get('@type','') == 'messageText':
            return message.get('content',{}).get('text',{}).get('text','')
    
class NewMailing(Dialog):
    """ Dialog of making a new mass mailing """

    def __init__(self, user, app):
        super().__init__(user,app)

    def start(self):
        self._answer('Create new mailing.')
        self.prepare_message()
    
    def prepare_message(self):
        self._answer('Prepare the message and paste or forward it here.')
        self.next = self.handle_mailing_message

    def handle_mailing_message(self, message):
        #print(dumps(message))
        input_message_content = self.app.conv.convert(message)
        if not input_message_content:
            self._answer('Message type not supported, try again')
            self.next = self.handle_mailing_message
            return
        self.result['message'] = self.app.conv.make_send_message(input_message_content)
        converted = self.app.conv.append_recepient_user(self.result['message'], self.user)
        self.app.bot.tdutil.send_message(converted)
        self._answer('This is how its gonna look. Is it ok?', ['OK', 'No'])
        self.next = self.confirm_message

    def confirm_message(self, message):
        text = self._get_text(message)
        if text == 'OK':
            self.enter_date()
        else:
            self.prepare_message()

    def enter_date(self):
        self._answer('Enter date of posting (like 17.06 16:20)')
        self.next = self.handle_date

    def handle_date(self, message):
        text = self._get_text(message)
        try:
            date = parser.parse(text)
            date = date.replace(tzinfo=ZoneInfo('Europe/Moscow'))
            self.result['date'] = date
        except ParserError as e:
            self._answer(f'Date format not recognized, try again')
            self.next = self.handle_date
            return

        self._answer(f'{date}\n\nIs that correct?', ['Yes','No'])

        self.next = self.confirm_date

    def confirm_date(self, message):

        #TODO: how do we save mailing post? sendMessage already has chat_id in it, but we 
        # want to store message decoupled from recepient info. SOLUTION: store input_message_content only
        text = self._get_text(message)
        if text == 'Yes':
            added = self.app.bot.save_mailing(json.dumps(self.result['message']), int(self.result['date'].timestamp()))
            if added:
                self._answer(f"Ok, planned on {self.result['date']}")
            else:
                self._answer('Not added, something went wrong')
            self.next = None
        else:
            self.enter_date()

