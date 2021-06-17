# Run this when new listeners are added to DB before oleg.py

from tdlib_client.client import Telegram
from staff.dba import OlegDBAdapter
import os

env = os.environ

api_id = env['api_id']
api_hash = env['api_hash']
database_encryption_key = env['listener_db_encryption_key']
files_directory = env['listener_files_directories']
db_name = env['db_name']
db_user = env['db_user']
db_password = env['db_password']
db_host = env['db_host']


dba = OlegDBAdapter(db_name,db_user,db_password,db_host)
listeners = dba.get_listeners()

print("Listener id to login? (and i'll try to login all ids that come after it)")
inp = input()


# TODO: TEST THIS


for i in range(inp,100000)
    if i in listeners:
        tg = Telegram(
            api_id = api_id,
            api_hash = api_hash,
            database_encryption_key = database_encryption_key,
            phone = inp,
            files_directory = os.path.join(files_directory, str(i)
        
        tg.login()
        
        tg.stop()
    else:
        print(f'{i} is not a Listener id')
        break

