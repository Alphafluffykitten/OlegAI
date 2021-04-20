import sqlite3
from sqlite3 import Error
from staff.olegtypes import *
import threading
from types import SimpleNamespace as NS
import json

class SQLiteDB:
    ''' sqlite3 db driver for threading '''
    def __init__(self,db_file:str=''):
        self.db_file = db_file
        self.conn = {} # DB connections pool for threads
        
    def _open_db_conn(self):
        """ creates a database connection to a SQLite database
            in current thread (if none) and stores it in self.conn[thread_ident] """
        
        thid = threading.get_ident()
        if not (str(thid) in self.conn):
            try:
                self.conn[str(thid)] = NS(conn     = sqlite3.connect(self.db_file,check_same_thread=False),
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
                      e.g. sql = 'INSERT INTO tbl (col1,col2,col3) VALUES (?,?,?)'
                           values = [1,2,3]
        '''
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(sql,values)
        conn.commit()
    
    def query(self,sql):
        ''' make SQL query in current thread conn '''
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        
        return rows
    
    
    
 

