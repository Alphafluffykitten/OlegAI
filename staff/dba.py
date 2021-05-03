from staff.pgsql import PostgresDB
from staff.olegtypes import *
import threading
from types import SimpleNamespace as NS
import json

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
        
    def register_user(self,tg_user_id):
        """ registers new user if there is no such tg_user_id, returns True if registered """
        
        user = self.get_user(tg_user_id = tg_user_id)
        
        #new_user = False
        if not user:
            sql = f'''
                INSERT INTO users (tg_user_id)
                 VALUES ({tg_user_id})
            '''
            self.db.push(sql)

            # get new user's user_id
            user = self.get_user(tg_user_id = tg_user_id)
            #new_user = True
            return user
        else:
            return False

    def get_user(self, user_id=None, tg_user_id=None):
        """ returns User object by OlegDB.users.id or OlegDB.users.tg_user_id """
        
        if user_id:
            sql = f'SELECT id, tg_user_id FROM users WHERE id={user_id} ORDER BY id DESC'
        elif tg_user_id:
            sql = f'SELECT id, tg_user_id FROM users WHERE tg_user_id={tg_user_id} ORDER BY id DESC'
        else:
            raise Exception("[ OlegDBAdapter ]: Neither user_id nor tg_user_id specified")
        
        rows = self.db.query(sql)
        if rows:
            user = User(id = rows[0][0],
                        tg_user_id = rows[0][1]
                       )
        else:
            user = None
            
        return user

    def get_user_ids(self):
        sql = 'SELECT id FROM users ORDER BY id ASC'
        rows = self.db.query(sql)
        ids = []
        for r in rows:
            ids.append(r[0])
        return ids

    def get_users(self):
        """ Gets users from OlegDB, newest last. Returns list of User objects """

        # for now it will just fetch all users from DB. If later we add some features to user,
        # we can add selectors for these features
        sql = f'SELECT id, tg_user_id FROM users'
        rows = self.db.query(sql)

        users = []
        for r in rows:
            users.append(User(
                id = r[0],
                tg_user_id = r[1]
            ))
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
        
        sql = f'''SELECT id, tg_msg_id, tg_channel_id, tg_timestamp, content_downloaded
                   FROM posts
                   {clauses}
                   ORDER BY id DESC {limit}'''
        rows = self.db.query(sql)
        
        posts = []
        for row in rows:
            posts.append(Post(id = row[0],
                              tg_msg_id = row[1],
                              tg_channel_id = row[2],
                              tg_timestamp = row[3],
                              content_downloaded = row[4]
                             )
                        )
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

        sql = f'SELECT id FROM posts {clauses} ORDER BY id ASC'
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
            sql = f'''INSERT INTO posts (tg_msg_id, tg_channel_id, tg_timestamp)
                    VALUES ({post.tg_msg_id}, {post.tg_channel_id}, '{post.tg_timestamp}')'''
            self.db.push(sql)

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

        sql = f'SELECT id, emoji, text from reactions'
        rows = self.db.query(sql)

        if not rows:
            raise Exception('[ OlegDBAdapter.get_reactions ]: No reactions found in DB')
            return None
        
        d = {}
        for r in rows:
            d[int(r[0])] = Reaction(
                id = int(r[0]),
                emoji = r[1],
                text = r[2]
            )
        
        return d

    def update_reaction(self,user_id,post_id,reaction):
        """ updates reaction info in DB or adds new entry """
        
        ur = self.get_user_reactions(user_id,post_id)

        # if user reaction already present, update it, otherwise write new
        if ur:
            ur = ur[0]
            sql = f'''
                UPDATE user_reactions
                SET reaction_id = {reaction}
                WHERE id = {ur.id}
            '''
        else:
            sql = f'''
                INSERT INTO user_reactions (user_id, internal_post_id, reaction_id)
                VALUES ({user_id}, {post_id}, {reaction})
            '''

        self.db.push(sql)


    def get_user_reactions(self, user_id=None, internal_post_id=None, learned:bool=None, limit=10000, order = 'DESC'):
        """
        gets OlegDB.user_reactions as list of olegtypes.UserReaction
        
        Args:
        user_id: int
        internal_post_id: int
        learned: bool
        limit: int. Default = 10000
        order: str. Default = 'desc' (newest first). Can be 'asc' or 'desc'
        """

        clauses = []
        if user_id:
            clauses.append(f'user_id = {user_id}\n')
        if internal_post_id:
            clauses.append(f'internal_post_id = {internal_post_id}\n')
        if learned is not None:
            clauses.append(f'learned = {bool(learned)}\n')

        clauses = ' AND\n'.join(clauses)
        if clauses:
            clauses = 'WHERE\n'+clauses

        if limit<0:
            limit = 10000

        order = order.upper()
        if order not in ('ASC', 'DESC'):
            order = 'DESC'

        sql = f'''
            SELECT id, user_id, internal_post_id, reaction_id, learned FROM user_reactions
            {clauses}
            ORDER BY id {order}
            LIMIT {limit}
        '''
        rows = self.db.query(sql)

        ur = []
        for r in rows:
            ur.append(
                UserReaction(
                    id = r[0],
                    user_id = r[1],
                    internal_post_id = r[2],
                    reaction_id = r[3],
                    learned = r[4]
                )
            )
        return ur

    def delete_post(self,post:Post):
        ''' deletes given post from OlegDB '''

        sql = f'DELETE FROM posts WHERE id = {post.id}'
        self.db.push(sql)

    def add_repost(self,post,user):
        ''' writes information about repost to DB '''

        repost = self.get_reposts(post.id,user.id)

        if not repost:
            sql = f'INSERT INTO reposts (internal_post_id,user_id) VALUES ({post.id}, {user.id})'
            self.db.push(sql)

    def get_reposts(self,post_id=None,user_id=None,limit=1000):
        ''' returns list of reposts for given post_id or user_id newest first'''

        clauses = []
        if post_id:
            clauses.append(f'internal_post_id = {post_id}')
        if user_id:
            clauses.append(f'user_id = {user_id}')
        
        if not clauses:
            raise Exception('[ OlegDBAdapter.get_reposts ]: Neither post_id nor user_id specified')

        clauses = '\nAND '.join(clauses)

        if limit < 0: limit = 1000
        
        sql = f'SELECT id, internal_post_id, user_id FROM reposts WHERE {clauses} ORDER BY id DESC LIMIT {limit}'

        rows = self.db.query(sql)

        reposts = []
        for row in rows:
            reposts.append(Repost(
                id = row[0],
                internal_post_id = row[1],
                user_id = row[2]
            ))
            
        return reposts

    def count_reposts(self,user):
        """ counts reposts sent to user """

        sql = f'SELECT COUNT(*) FROM {Repost.table_name} WHERE user_id = %s'
        rows = self.db.query(sql,[user.id])
        return rows[0][0]

    def set_reactions_learned(self, ur):
        """ sets learned=1 in OlegDB.posts for ids """

        ids = []
        for r in ur:
            ids.append(r.id)

        if ids and isinstance(ids,list):
            ids = list(map(str,ids))
            ids = ', '.join(ids)
            sql = f'''
            UPDATE user_reactions
            SET learned = true
            WHERE id IN ({ids})
            '''
            self.db.push(sql)

    def get_channel(self, id = None, tg_channel_id = None, listening = True, name = None):
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

        sql = f'SELECT id, tg_channel_id, listening, name FROM channels {clauses} ORDER BY id ASC'
        #print(sql)
        rows = self.db.query(sql)
        channels = []
        for r in rows:
            channels.append(
                Channel(
                    id = r[0],
                    tg_channel_id = r[1],
                    listening = r[2],
                    name = r[3]
                )
            )
        return channels

    def add_channel(self, channel):
        """
        Adds new channel to OlegDB or updates info on existing (except id)
        
        Returns:
        Channel that was added or updated
        """

        channels = self.get_channel(tg_channel_id = channel.tg_channel_id)
        if not channels:
            sql = f'INSERT INTO channels (tg_channel_id, listening, name) VALUES (%s, %s, %s)'
        else:
            sql = f'''
            UPDATE channels
            SET
            tg_channel_id = %s,
            listening = %s,
            name = %s
            WHERE
            id = {channels[0].id}
            '''

        self.db.push(sql,[channel.tg_channel_id, channel.listening, channel.name])

        added = self.get_channel(tg_channel_id = channel.tg_channel_id)[0]
        
        return added

        