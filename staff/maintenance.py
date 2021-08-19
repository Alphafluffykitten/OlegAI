from staff.dba import OlegDBAdapter
from staff.utils import TDLibUtils
import os

def get_listeners():
    env = os.environ

    db_name = env['db_name']
    db_user = env['db_user']
    db_password = env['db_password']
    db_host = env['db_host']


    dba = OlegDBAdapter(db_name,db_user,db_password,db_host)
    dba.start()
    listeners = dba.get_listeners()
    dba.stop()

    return listeners

def get_tdutil(l_id, phone):
    env = os.environ

    api_id = env['api_id']
    api_hash = env['api_hash']
    database_encryption_key = env['listener_db_encryption_key']
    files_directory = env['listener_files_directories']

    return TDLibUtils(
        api_id = api_id,
        api_hash = api_hash,
        phone = phone,
        database_encryption_key = database_encryption_key,
        files_directory = os.path.join(files_directory, str(l_id))
    )

def show_listeners(ls):
    for l in ls:
        print(f'{l} - {ls[l].phone}')

def destroy_tdlib(tdutil):
    
    print('Logging in')
    tdutil.start()
    print('Destroying TDLib')
    res = tdutil.destroy()
    if res: print('Done')
    tdutil.stop()

def login_tdlib(tdutil):
    print('Logging in')
    tdutil.start()
    print('Getting chat list')
    chats = tdutil.get_all_chats_list()
    print(chats)
    print('Done')
    tdutil.stop()
