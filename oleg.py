import logging
from staff.utils import OlegApp
import threading, signal
import os

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     level=logging.DEBUG)

stopit = threading.Event()
def got_ipt(signo,frame):
    stopit.set()
for sig in ('TERM', 'HUP', 'INT'):
    signal.signal(getattr(signal, 'SIG'+sig), got_ipt)

env = os.environ

app = OlegApp(
    db_name = env['db_name'],
    db_user = env['db_user'],
    db_password = env['db_password'],
    db_host = env['db_host'],
    bot_token = env['bot_token'],
    api_id = env['api_id'],
    api_hash = env['api_hash'],
    admin_phone = env['admin_phone'],
    admin_db_encryption_key = env['admin_db_encryption_key'],
    bot_db_encryption_key = env['bot_db_encryption_key'],
    admin_files_directory = env['admin_files_directory'],
    bot_files_directory = env['bot_files_directory'],
    salt = env['salt']
)


app.start()

print('OlegApp started, press ctrl-c to shutdown gracefully')
while not stopit.is_set():
    stopit.wait()

app.stop()