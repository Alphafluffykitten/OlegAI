import psycopg2
from psycopg2 import Error
from staff.olegtypes import *
import threading
from types import SimpleNamespace as NS
import json

class PostgresDB:
    ''' postgres db driver. Maintains dict of connections, one per thread '''

    def __init__(self, dbname, user, password, host):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.conn = {} # DB connections pool for threads
        
    def _open_db_conn(self):
        """
        creates a database connection to a SQLite database
        in current thread (if none) and stores it in self.conn[thread_ident]
        """
        
        thid = threading.get_ident()
        if not (str(thid) in self.conn):
            try:
                self.conn[str(thid)] = NS(
                    conn = psycopg2.connect(
                        dbname = self.dbname,
                        user = self.user,
                        password = self.password,
                        host = self.host
                    ),
                    is_alive = True
                )
            except Error as e:
                print(e)
                
    def get_conn(self):
        """ returns a connection from the pool for current thread or tries to make one, or returns None """
        thid = threading.get_ident()
        
        conn = None
        if not ((str(thid) in self.conn) and (self.conn[str(thid)].is_alive)):
            self._open_db_conn()
            
        conn = self.conn[str(thid)].conn
        
        return conn
        
    def close(self):
        """ close all connections """
        for i in self.conn:
            if self.conn[i].is_alive:
                self.conn[i].is_alive = False
                self.conn[i].conn.close()

    def push(self,sql, values:list=[]):
        '''
        make SQL injection in current thread conn
        sql: str. SQL injection
        values: list. List of values for parametrized SQL query,
                      e.g. sql = 'INSERT INTO tbl (col1,col2,col3) VALUES (%s,%s,%s)'
                           values = [1,2,3]
        '''
        conn = self.get_conn()
        cur = conn.cursor()
        try:
            cur.execute(sql,values)
        except Error as e:
            print(f'SQL caused error. \nSQL:\n{sql}')
            print(f'Error:\n{e}')
            raise e
        conn.commit()
        cur.close()
    
    def query(self,sql):
        ''' make SQL query in current thread conn '''
        conn = self.get_conn()
        cur = conn.cursor()
        rows = []
        try:
            cur.execute(sql)
            rows = cur.fetchall()
        except Error as e:
            print(f'SQL caused error. \nSQL:\n{sql}')
            print(f'Error:\n{e}')
            raise e
        conn.commit()
        cur.close()
        
        return rows
    
    
    
 

