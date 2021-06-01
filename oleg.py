import logging
from staff.utils import OlegApp
import threading, signal
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

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
    listener_db_encryption_key = env['listener_db_encryption_key'],
    bot_db_encryption_key = env['bot_db_encryption_key'],
    listener_files_directories = env['listener_files_directories'],
    bot_files_directory = env['bot_files_directory'],
    salt = env['salt'],
    logs_dir = env['logs_dir'],
    admin_user_id = env['admin_user_id'],
    nn_user_lpv_len = env['nn_user_lpv_len'],
    nn_post_lpv_len = env['nn_post_lpv_len'],
    nn_channel_lpv_len = env['nn_channel_lpv_len'],
    nn_hidden_layer = env['nn_hidden_layer'],
    nn_new_reactions_threshold = env['nn_new_reactions_threshold'],
    nn_learning_timeout = env['nn_learning_timeout'],
    nn_full_learn_threshold = env['nn_full_learn_threshold'],
    nn_closest_shuffle = env['nn_closest_shuffle'],
    max_listener_channels = env['max_listener_channels']
)


app.start()

print('OlegApp started, press ctrl-c to shutdown gracefully')
while not stopit.is_set():
    stopit.wait()

app.stop()