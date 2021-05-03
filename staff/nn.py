import torch
from torch import nn, tensor
import time
import threading
from staff.olegtypes import *

class DotProduct(nn.Module):
    """ Dot product model for learning """
    def __init__(self, user_emb, user_bias, post_emb, post_bias, idx2user, idx2post):
        """
        Args:
        user_emb, user_bias, post_emb, post_bias are tuples denoting dimensions of nn.Embedding
        idx2user, idx2post - lists where elements are ids of users (posts) in OlegDB,
            indexes equal to their indexes in embedding matrices
        """
        super(DotProduct, self).__init__()

        self.sig = nn.Sigmoid()

        self.u = nn.Embedding(*user_emb)
        self.u_bias = nn.Embedding(*user_bias)

        self.p = nn.Embedding(*post_emb)
        self.p_bias = nn.Embedding(*post_bias)

        # attach vocabs to model
        self.idx2post = idx2post
        self.idx2user = idx2user
        # user and post vocabulars:
        # keys are DB user_ids, values are idxs in embedding matrices 
        self.user2idx = {o:i for i,o in enumerate(idx2user)}
        self.post2idx = {o:i for i,o in enumerate(idx2post)}

    def forward(self, x):
        users = self.u(x[:,0])
        posts = self.p(x[:,1])
        res = (users * posts).sum(dim=1,keepdim=True)
        res += self.u_bias(x[:,0]) + self.p_bias(x[:,1])
        res = res.squeeze(1)
        return self.sig(res)*1.05

class DataLoader():
    """ takes list of UserReactions and produces an iterator over batches """

    # vocabularize reaction_ids (get regression values off the reactions)
    reaction2value = {1: 1, 2: 0}

    def __init__(self,ur,bs,user2idx,post2idx):
        """
        Args:
        ur (list:UserReaction): list of user reactions
        bs (int): batch size
        user2idx (dict): vocab translating user_id from OlegDB to user embedding idx
        post2idx (dict): vocab translating post_id from OlegDB to post embedding idx 
        """

        self.bs = bs
        self.ur = ur

        self.cur = 0 # points on current idx in ur

        self.user2idx = user2idx
        self.post2idx = post2idx


    def __iter__(self):
        return self

    def __next__(self):
        if self.cur > len(self.ur)-1:
            self.cur = 0
            raise StopIteration

        x_stack = []
        y_stack = []
        for i in range(self.bs):
            if self.cur <= len(self.ur)-1:
                user_voc = self.user2idx[self.ur[self.cur].user_id] # vocabularized user id
                post_voc = self.post2idx[self.ur[self.cur].internal_post_id]
                reac_voc = self.reaction2value[self.ur[self.cur].reaction_id]
                x_stack.append(tensor([int(user_voc),int(post_voc)]))
                y_stack.append(tensor(reac_voc,dtype=torch.float32))
            else:
                break
            self.cur += 1

        x_stack = torch.stack(x_stack)
        y_stack = torch.stack(y_stack)

        return x_stack, y_stack

class InferDataLoader():
    """ data loader for inferring """

    def __init__(self,posts,user,user2idx,post2idx):
        self.posts = posts
        self.user = user

        # get post ids
        #idx2post = []
        #for post in posts:
        #    idx2post.append(post.id)

        #idx2post.sort()
        # idx2user always consists of 1 user here
        
        self.user2idx = user2idx
        self.post2idx = post2idx

        
    def get_data(self):
        x_stack = []
        for p in self.posts:
            # user is always unique here
            x_stack.append(tensor([self.user2idx[self.user.id],self.post2idx[p.id]]))
        x_stack = torch.stack(x_stack)
        self.x_stack = x_stack

        return x_stack



class OlegNN():
    """
    neural net functions for OlegAI
    """

    lr = 100        # learning rate
    bs = 512        # batch size 
    incremental_epochs = 10     # epochs for incremental training on new reactions
    full_learn_epochs = 35      # epochs to train on new learning dataset

    def __init__(
        self,
        lpv_len,                    # length of latent parameters vector
        nn_new_reactions_threshold, # how many new reactions to start incremental learning cycle
        nn_learning_timeout,        # min interval between learning cycles, seconds
        nn_full_learn_threshold,    # how many incremental learning cycles to start a full learning cycle
    ):
        self.lpv_len = int(lpv_len)         
        self.new_reactions_threshold = int(nn_new_reactions_threshold)
        self.learning_timeout = int(nn_learning_timeout)
        self.full_learn_threshold = int(nn_full_learn_threshold)

        self.new_reactions_counter = 0      # new reactions since last learning cycle
        self.inc_cycles = 0                 # how many incremental cycles have passed since last full cycle
        self.last_learning_cycle = 0        # when was last learning cycle
        
        self.learning = False
        self.losses = []

    def start(self):
        # run learning in serial mode, will instantiate new model
        self.learn_data(full=True)


    def init_model(self):
        """ instantiates new model with random embeddings """
        
        # make vocab for posts
        idx2post = self.app.dba.get_post_ids()
        idx2user = self.app.dba.get_user_ids()
        self.model = DotProduct(
            (len(idx2user),self.lpv_len),
            (len(idx2user),1),
            (len(idx2post),self.lpv_len),
            (len(idx2post),1),
            idx2user,
            idx2post
        )


    def closest(self,posts,user):
        """ returns post closest to user """

        dl = InferDataLoader(posts,user,self.model.user2idx,self.model.post2idx)

        preds = self.model.forward(dl.get_data())
        # add some shuffle to result
        preds = self.shuffle(preds,amount=0.01)
        self.inf_preds = preds
        topidx = torch.argmax(preds).item()
        v_post_id = dl.x_stack[topidx][1]

        for post in posts:
            if post.id == self.model.idx2post[v_post_id]:
                return post

    def get_init_emb(self):
        """
        returns randomly initialized
        embedding vector (for user or post doesnt matter) as torch.Tensor
        """

        return torch.randn(self.lpv_len)

    def got_new_reaction(self):
        """
        Checks conditions, if it has to run incremental learning, or full learning, and runs it
        """

        now = time.time()

        self.new_reactions_counter += 1
        # if enough reactions and timeout is ok, try to run learning
        if (
            self.new_reactions_counter >= self.new_reactions_threshold and
            now - self.last_learning_cycle >= self.learning_timeout
        ):
            # if not learning now, run learning process
            if not self.learning:
                self.new_reactions_counter = 0
                self.last_learning_cycle = now

                # every self.inc_cycles it runs full learning cycle
                #print(f'inc cycles {self.inc_cycles} >= full learn thr {self.full_learn_threshold}')
                if self.inc_cycles >= self.full_learn_threshold:
                    # full cycle
                    #print('run FULL cycle')
                    self.init_model()
                    threading.Thread(target=self.learn_data, kwargs = {'full': True}).start()
                    self.inc_cycles = 0
                else:
                    # incremental cycle
                    #print('run INC cycle')
                    threading.Thread(target=self.learn_data).start()
                    self.inc_cycles += 1


    def learn_data(self, full = False):
        """
        instantiate new learning model,
        learn data, write updated embeddings to DB

        Args:
        full (bool): if set, will learn full dataset of user reactions
        """
        
        self.learning = True

        #DEBUG
        st = time.time()
        self.st = st

        # get reactions
        if full:
            ur = self.app.dba.get_user_reactions()
            self.init_model()
        else:
            ur = self.app.dba.get_user_reactions(learned=0)
            # add 96% old reactions to dataset or 4 batches
            ur = ur + self.app.dba.get_user_reactions(learned=1,limit=max(len(ur)*30,self.bs*4-len(ur)))
        
        if not ur:
            self.learning = False
            return

        #print(f'{(time.time()-st):.4f} got {len(ur)} reactions')
        
        # DataLoader should provide batches like x.shape = (bs,2) y.shape = (bs,1)
        dl = DataLoader(ur,self.bs, self.model.user2idx, self.model.post2idx)
        self.dl = dl

        #print(f'{(time.time()-st):.4f} got dl')

        if full:
            epochs = self.full_learn_epochs
        else:
            # prelearned embeddings already are in memory
            epochs = self.incremental_epochs

        loss_func = nn.MSELoss()
        self._learn(dl, self.model, loss_func, epochs)
        #print(f'{(time.time()-st):.4f} finished learning')

        # set learned=1 for reactions in dl
        self.app.dba.set_reactions_learned(ur)

        if full:
            self._update_rest_emb()
            #print(f'{(time.time()-self.st):.4f} updated rest embs')
        #print(f'{(time.time()-st):.4f} updated embs')

        self.learning = False

        #print(f'Learn time  {(time.time()-st):.4f} s')

    def _learn(self,dl,model,loss_func,epochs=1):
        
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0)

        for e in range(epochs):
            for d in dl:
                model.zero_grad()
                preds = model.forward(d[0])
                self.preds = preds
                loss = loss_func(preds,d[1])
                self.losses.append(loss.item())

                loss.backward()
                optim.step()

    def _update_rest_emb(self):
        """
        updates post embeddins that were not in dataset
        (sets same embeddings as in posts from learned channels)
        """
        
        # get unique channels from dataset
        posts = self.app.dba.get_posts(dataset = 'include')
        u_channels = []
        for p in posts:
            u_channels.append(p.tg_channel_id)
        u_channels = list(set(u_channels))

        self.model.p.weight.detach_()
        self.model.p_bias.weight.detach_()

        for c in u_channels:
            # get posts from this channel that were in training dataset
            ref_posts = self.app.dba.get_posts(tg_channel_id = c, dataset='include', have_content=True, limit=1000)
            channel_emb, channel_bias = self.mean_of_posts(ref_posts)
            # get all posts from this channel that were not in learning dataset
            non_ds_posts = self.app.dba.get_posts(tg_channel_id = c, dataset = 'exclude', have_content=True)
            for p in non_ds_posts: 
                self.model.p.weight[self.post2idx(p.id)] = self.shuffle(channel_emb)
                self.model.p_bias.weight[self.post2idx(p.id)] = channel_bias
        self.model.p.requires_grad_(True)
        self.model.p_bias.requires_grad_(True)

    def post2idx(self,post):
        """ takes Post obj or internal_post_id and returns its embedding idx """
        if isinstance(post, Post):
            return self.model.post2idx[post.id]
        else:
            return self.model.post2idx[post]
    
    def shuffle(self, t:torch.Tensor, amount=0.05):
        """ adds shuffle to input tensor """
        return t + amount * torch.randn(t.shape)

    def get_embs(self,post_ids):
        """
        takes list of internal post ids as argument

        Returns tuple of tensors of embs and biases for these posts
        tensors have shape (len(post_ids), lpv_len)
        """

        v_ids = []
        for i in post_ids:
            v_ids.append(self.post2idx(i))
        return (
            self.model.p.weight[v_ids].clone().detach(),
            self.model.p_bias.weight[v_ids].clone().detach()
        )

    def add_emb(self,where,post=None,user=None,emb=None,bias=None):
        """ adds new embedding to embedding matrix """

        # block until learning cycle done
        while self.learning:
            time.sleep(0.1)

        self.learning = True

        emb = emb.unsqueeze(0)
        bias = bias.unsqueeze(0)

        # add emb to post embeddings
        if where == 'post':
            if post.id not in self.model.post2idx:
                emb_rows = self.model.p.weight.shape[0]

                # add row to emb and bias
                self.model.p = nn.Embedding.from_pretrained(
                    torch.cat((self.model.p.weight.detach(), emb), dim=0)
                )
                self.model.p.weight.requires_grad_(True)
                self.model.p_bias = nn.Embedding.from_pretrained(
                    torch.cat((self.model.p_bias.weight.detach(), bias), dim=0)
                )
                self.model.p_bias.weight.requires_grad_(True)

                # add value to vocab
                self.model.idx2post.append(post.id)
                self.model.post2idx[post.id] = emb_rows

        # add emb to user embeddings
        elif where == 'user':
            if user.id not in self.model.user2idx:

                emb_rows = self.model.u.weight.shape[0]

                # add row to emb and bias
                self.model.u = nn.Embedding.from_pretrained(
                    torch.cat((self.model.u.weight.detach(), emb), dim=0)
                )
                self.model.u.weight.requires_grad_(True)

                self.model.u_bias = nn.Embedding.from_pretrained(
                    torch.cat((self.model.u_bias.weight.detach(), bias), dim=0)
                )
                self.model.u_bias.weight.requires_grad_(True)

                # add value to vocab
                self.model.idx2user.append(user.id)
                self.model.user2idx[user.id] = emb_rows
        else:
            raise Exception('[ OlegNN.add_emb ]: where must be "post" or "user"')

        self.learning = False

    def mean_of_posts(self, posts):
        """ returns mean embedding and bias of posts """

        post_ids = []
        for p in posts:
            post_ids.append(p.id)
        embs, biases = self.get_embs(post_ids)

        emb = embs.mean(dim=0)
        bias = biases.mean(dim=0)
        return emb, bias
    
    def get_max_bias(self,k=100):
        """
        Returns list of post indexes with max bias, in descending order
        Args:
        k (int): how many top elements to get
        """

        topk = torch.topk(self.model.p_bias.weight,k,dim=0,sorted=True)
        v_idxs = topk.indices.squeeze().tolist()

        post_idxs = []
        for i in v_idxs:
            post_idxs.append(self.model.idx2post[i])
        
        return post_idxs

