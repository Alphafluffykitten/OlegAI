from staff.pgsql import PostgresDB
from staff.olegtypes import *
import threading
from types import SimpleNamespace as NS
import json
import time

class OlegDBAdapter():
    """ DB adapter for PostgreSQL. Holds info about OlegDB structure and provides objects from DB to clients """
    
    def __init__(self, dbname, user, password, host):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host

    def start(self):
        self.db = PostgresDB(self.dbname, self.user, self.password, self.host)
        
    def stop(self):
        self.db.close()
        
    def register_user(self, tg_user_id, username):
        """ registers new user if there is no such tg_user_id, returns True if registered """
        
        user = self.get_user(tg_user_id = tg_user_id)
        
        if not user:
            sql = f'''
                INSERT INTO {User.table_name} (tg_user_id, username, timestamp)
                 VALUES (%s, %s, %s)
            '''
            self.db.push(sql,[tg_user_id, username, int(time.time())])

            # get new user
            user = self.get_user(tg_user_id = tg_user_id)
            return user
        else:
            return False

    def get_user(self, user_id=None, tg_user_id=None):
        """ returns User object by OlegDB.users.id or OlegDB.users.tg_user_id """
        
        if user_id:
            sql = f'SELECT {", ".join(User.cols)} FROM {User.table_name} WHERE id={user_id} ORDER BY id DESC'
        elif tg_user_id:
            sql = f'SELECT {", ".join(User.cols)} FROM {User.table_name} WHERE tg_user_id={tg_user_id} ORDER BY id DESC'
        else:
            raise Exception("[ OlegDBAdapter ]: Neither user_id nor tg_user_id specified")
        
        rows = self.db.query(sql)
        if rows:
            r = rows[0]
            kwargs = {colname:r[idx] for idx,colname in enumerate(User.cols)}
            user = User(**kwargs)
        else:
            user = None
            
        return user

    def get_user_ids(self):
        sql = f'SELECT id FROM {User.table_name} ORDER BY id ASC'
        rows = self.db.query(sql)
        ids = []
        for r in rows:
            ids.append(r[0])
        return ids

    def get_users(self):
        """ Gets users from OlegDB, newest last. Returns list of User objects """

        # for now it will just fetch all users from DB. If later we add some features to user,
        # we can add selectors for these features
        sql = f'SELECT {", ".join(User.cols)} FROM users'
        rows = self.db.query(sql)

        users = []
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(User.cols)}
            users.append(User(**kwargs))
        return users

    def get_posts(
        self, ids = [], exc_ids = [], dataset = None, tg_msg_id=None, tg_channel_id=None, have_content=None,
        tg_timestamp_range=(), limit=None, except_user=None):
        """
        Gets posts from OlegDB, newest first
        
        Args:
        ids (list:int, optional): post ids to look among
        exc_ids (list:int, optional): post ids to exclude
        dataset (str, optional): can be 'include', 'exclude' or None. If 'include' then only selects from posts
            that present in user_reactions table. If 'exclude', excludes these posts from select
        tg_msg_id (int, optional): tg_msg_id to search for
        tg_channel_id (int,optional): tg_channel_id to search for
        have_content (bool, optional): if flagged, searches only for content_downloaded=1
        tg_timestamp_range (tuple, optional): if passed, searches only for timestamps within range
        limit (int, optional): default = 1000, max = 1000
        except_user (int, optional): if passed, will return only those posts which were not reposted to this user

        Returns:
        list of Post objects, newest first
        """
        
        clauses = []
        if ids :
            ids = list(map(str,ids))
            ids = ', '.join(ids)
            clauses.append(f'id IN ({ids})')

        if exc_ids:
            ids = list(map(str,exc_ids))
            ids = ', '.join(ids)
            clauses.append(f'id NOT IN ({ids})')

        if tg_msg_id:
            clauses.append(f'tg_msg_id = {tg_msg_id}')

        if tg_channel_id:
            clauses.append(f'tg_channel_id = {tg_channel_id}')

        if have_content is not None:
            clauses.append(f'content_downloaded = {bool(have_content)}')

        if tg_timestamp_range:
            clauses.append(f'tg_timestamp BETWEEN {tg_timestamp_range[0]} AND {tg_timestamp_range[1]}')

        if except_user:
            clauses.append(f'''
                id NOT IN (
                    SELECT internal_post_id FROM reposts WHERE user_id = {except_user}
                )
            ''')

        if dataset:
            dataset = dataset.lower()
        if dataset == 'include':
            clauses.append(f'''
                id IN (
                    SELECT internal_post_id FROM user_reactions
                )
            ''')
        elif dataset == 'exclude':
            clauses.append(f'''
                id NOT IN (
                    SELECT internal_post_id FROM user_reactions
                )
            ''')

        if clauses:
            clauses = '\nAND '.join(clauses)
            clauses = 'WHERE\n' + clauses
        else:
            clauses = ''

        if limit is not None:
            if limit <= 0:
                limit = 1000
            limit = f'LIMIT {limit}'
        else:
            limit = ''
        
        sql = f'''SELECT {", ".join(Post.cols)}
                   FROM {Post.table_name}
                   {clauses}
                   ORDER BY id DESC {limit}'''
        rows = self.db.query(sql)
        
        posts = []
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(Post.cols)}
            posts.append(Post(**kwargs))
        return posts

    def get_post_ids(self,content_downloaded=None):
        """ gets list of post ids from OlegDB """

        clauses = []
        if content_downloaded is not None:
            clauses.append(f'content_downloaded = {bool(content_downloaded)}')

        if clauses:
            clauses = ' AND\n'.join(clauses)
            clauses = f'WHERE {clauses}'
        else:
            clauses = ''

        sql = f'SELECT id FROM {Post.table_name} {clauses} ORDER BY id ASC'
        #print(sql)
        rows = self.db.query(sql)
        ids = []
        for r in rows:
            ids.append(r[0])
        return ids

    def add_post(self, post: Post):
        """
        adds post to DB 

        Args:
        post: olegtypes.Post object (without id field)
        """

        posts = self.get_posts(tg_msg_id = post.tg_msg_id, tg_channel_id = post.tg_channel_id)

        if not posts:
            # insert new post
            sql = f'''INSERT INTO {Post.table_name} (tg_msg_id, tg_channel_id, tg_timestamp, timestamp)
                    VALUES (%s, %s, %s, %s)'''
            self.db.push(sql,[post.tg_msg_id, post.tg_channel_id, post.tg_timestamp, int(time.time())])

            # get newly added post
            new_post = self.get_posts(tg_msg_id = post.tg_msg_id, tg_channel_id = post.tg_channel_id)
            if not new_post:
                raise Exception(f'[ OlegDBAdapter ]: Couldn\'t add new post to DB')

            return new_post[0]

    def set_content_downloaded(self, post: Post, cd=1):
        """ sets content_downloaded=1 """
        
        sql = f'''
            UPDATE posts
            SET content_downloaded = {bool(cd)}
            WHERE
                id = {post.id}
        '''
        self.db.push(sql)

    def get_reactions(self):
        """ returns reaction types as dict of olegtypes.Reaction objects """

        sql = f'SELECT {", ".join(Reaction.cols)} from {Reaction.table_name}'
        rows = self.db.query(sql)

        if not rows:
            raise Exception('[ OlegDBAdapter.get_reactions ]: No reactions found in DB')
            return None
        
        d = {}
        for r in rows:
            kwargs = {colname:r[idx] for idx, colname in enumerate(Reaction.cols)}
            d[r[0]] = Reaction(**kwargs)
        
        return d

    def update_reaction(self,user_id,post_id,reaction):
        """ updates reaction info in DB or adds new entry """
        
        ur = self.get_user_reactions(user_id,post_id)

        # if user reaction already present, update it, otherwise write new
        if ur:
            ur = ur[0]
            sql = f'''
                UPDATE {UserReaction.table_name}
                SET
                reaction_id = %s,
                timestamp = %s
                WHERE id = {ur.id}
            '''
            self.db.push(sql,[reaction, int(time.time())])
        else:
            sql = f'''
                INSERT INTO {UserReaction.table_name} (user_id, internal_post_id, reaction_id, timestamp)
                VALUES (%s, %s, %s, %s)
            '''
            self.db.push(sql,[user_id, post_id, reaction, int(time.time())])


    def get_user_reactions(
        self,
        user_id=None,
        internal_post_id=None,
        learned:bool=None,
        limit=None,
        order = 'RANDOM',
        with_channels=False
    ):
        """
        gets OlegDB.user_reactions as list of olegtypes.UserReaction
        
        Args:
        user_id: int
        internal_post_id: int
        learned: bool
        limit: int. Default = 10000
        order: str. Default = 'random' (newest first). Can be 'asc' or 'desc' or 'random'
        with_channels: bool. If True, joins OlegDB.posts.tg_channel_id
        """

        clauses = []
        if user_id:
            clauses.append(f'ur.user_id = {user_id}\n')
        if internal_post_id:
            clauses.append(f'ur.internal_post_id = {internal_post_id}\n')
        if learned is not None:
            clauses.append(f'ur.learned = {bool(learned)}\n')

        clauses = ' AND\n'.join(clauses)
        if clauses:
            clauses = 'WHERE\n'+clauses


        if limit is not None:
            limit= f'LIMIT {limit}'
        else:
            limit = ''

        order = order.upper()
        if order not in ('ASC', 'DESC', 'RANDOM'):
            order = 'RANDOM'
        if order == 'RANDOM':
            order = 'RANDOM()'
        elif order == 'ASC':
            order = 'ur.id ASC'
        elif order == 'DESC':
            order = 'ur.id DESC'
        
        suf_cols = []
        cols = UserReaction.cols.copy()
        for i in range(len(cols)):
            suf_cols.append(f'ur.{cols[i]}')

        join_clause = ''
        if with_channels:
            join_clause = f'LEFT JOIN {Post.table_name} p ON ur.internal_post_id = p.id'
            cols.append('tg_channel_id')
            suf_cols.append('p.tg_channel_id')

        sql = f'''
            SELECT {", ".join(suf_cols)} FROM {UserReaction.table_name} ur
            {join_clause}
            {clauses}
            ORDER BY {order}
            {limit}
        '''

        rows = self.db.query(sql)
        ur = []
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(cols)}
            ur.append(UserReaction(**kwargs))
        return ur

    def add_repost(self, post, user):
        ''' writes information about repost to OlegDB '''

        repost = self.get_reposts(post.id,user.id)

        if not repost:
            sql = f'INSERT INTO {Repost.table_name} (internal_post_id, user_id, timestamp) VALUES (%s, %s, %s)'
            self.db.push(sql, [post.id, user.id, int(time.time())])

    def get_reposts(self,post_id=None,user_id=None,limit=1000):
        ''' returns list of reposts for given post_id or user_id newest first '''

        clauses = []
        if post_id:
            clauses.append(f'internal_post_id = {post_id}')
        if user_id:
            clauses.append(f'user_id = {user_id}')
        
        if not clauses:
            raise Exception('[ OlegDBAdapter.get_reposts ]: Neither post_id nor user_id specified')

        clauses = '\nAND '.join(clauses)

        if limit < 0: limit = 1000
        
        sql = f'SELECT {", ".join(Repost.cols)} FROM {Repost.table_name} WHERE {clauses} ORDER BY id DESC LIMIT {limit}'

        rows = self.db.query(sql)

        reposts = []
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(Repost.cols)}
            reposts.append(Repost(**kwargs))
            
        return reposts

    def count_reposts(self,user):
        """ counts reposts sent to user """

        sql = f'SELECT COUNT(*) FROM {Repost.table_name} WHERE user_id = %s'
        rows = self.db.query(sql,[user.id])
        return rows[0][0]

    def set_reactions_learned(self, ur):
        """ sets learned=1 in OlegDB.user_reactions for list of UserReaction """

        ids = []
        for r in ur:
            ids.append(r.id)

        if ids and isinstance(ids,list):
            ids = list(map(str,ids))
            ids = ', '.join(ids)
            sql = f'''
            UPDATE {UserReaction.table_name}
            SET learned = true
            WHERE id IN ({ids})
            '''
            self.db.push(sql)

    def get_channels(self, id = None, tg_channel_id = None, listening = True, name = None):
        """ gets list of channels from OlegDB """

        clauses = []
        if id is not None:
            clauses.append(f'id = {id}')
        
        if tg_channel_id is not None:
            clauses.append(f'tg_channel_id = {tg_channel_id}')

        if listening is not None:
            clauses.append(f'listening = {bool(listening)}')

        if name is not None:
            clauses.append(f'name = {name}')
        
        if clauses:
            clauses = ' AND\n'.join(clauses)
            clauses = f'WHERE {clauses}'
        else:
            clauses = ''

        sql = f'SELECT {", ".join(Channel.cols)} FROM {Channel.table_name} {clauses} ORDER BY id ASC'
        #print(sql)
        rows = self.db.query(sql)
        channels = []
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(Channel.cols)}
            channels.append(Channel(**kwargs))
        return channels

    def get_tg_channel_ids(self):
        """ Returns list of channels' tg_channel_id """

        sql = f'SELECT tg_channel_id FROM {Channel.table_name} ORDER BY id ASC'
        rows = self.db.query(sql)
        ids = []
        for r in rows:
            ids.append(r[0])
        return ids


    def add_channel(self, channel):
        """
        Adds new channel to OlegDB or updates info on existing (except id)
        
        Returns:
        Channel that was added or updated
        """

        channels = self.get_channels(tg_channel_id = channel.tg_channel_id)
        if not channels:
            sql = f'INSERT INTO channels (tg_channel_id, listening, name, listener_id, timestamp) VALUES (%s, %s, %s, %s, %s)'
        else:
            sql = f'''
            UPDATE channels
            SET
            tg_channel_id = %s,
            listening = %s,
            name = %s,
            listener_id = %s,
            timestamp = %s
            WHERE
            id = {channels[0].id}
            '''

        self.db.push(sql,[channel.tg_channel_id, channel.listening, channel.name, channel.listener_id, int(time.time())])

        added = self.get_channels(tg_channel_id = channel.tg_channel_id)[0]
        
        return added

    def get_listeners(self):
        """ returns dict of Listeners where keys are Listeners' ids """

        sql = f'SELECT {", ".join(Listener.cols)} FROM {Listener.table_name} ORDER BY id ASC'

        rows = self.db.query(sql)
        listeners = {}
        for r in rows:
            kwargs = {colname:r[idx] for idx,colname in enumerate(Listener.cols)}
            listeners[kwargs['id']] = Listener(**kwargs)
        return listeners
        
    def get_listeners_volume(self):
        """ returns dict where keys are listeners ids, values are this listeners channels qty """

        sql = f'''
        SELECT
            L.ID,
            COUNT(C.LISTENER_ID)
        FROM LISTENERS L
        LEFT JOIN CHANNELS C ON C.LISTENER_ID = L.ID
        GROUP BY L.ID
        '''

        rows = self.db.query(sql)
        res = {}
        for r in rows:
            res[r[0]] = r[1]
        return res