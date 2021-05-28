import torch
from torch import nn, tensor
import random
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

class UserPostChannelNN(nn.Module):
    """ Model that distinguishes channels and posts """

    def __init__(
        self,
        user_emb,
        post_emb,
        channel_emb,
        idx2user, idx2post, idx2channel,
        n_hidden
    ):
        """
        Args:
        user_emb, post_emb, channel_emb are tuples denoting dimensions of nn.Embedding
        idx2user, idx2post, idx2channel - lists where elements are ids of users/posts/channels in OlegDB,
            indexes equal to their indexes in embedding matrices
        n_hidden - number of activations in hidden layer
        """

        super(UserPostChannelNN, self).__init__()

        self.sig = nn.Sigmoid()

        self.u = nn.Embedding(*user_emb)
        self.p = nn.Embedding(*post_emb)
        self.c = nn.Embedding(*channel_emb)

        self.layers = nn.Sequential(
            nn.Linear(user_emb[1]+post_emb[1]+channel_emb[1], n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

        self.idx2user = idx2user
        self.idx2post = idx2post
        self.idx2channel = idx2channel

        self.user2idx = {o:i for i,o in enumerate(idx2user)}
        self.post2idx = {o:i for i,o in enumerate(idx2post)}
        self.channel2idx = {o:i for i,o in enumerate(idx2channel)}

    def forward(self, x):
        u_emb = self.u(x[:,0])
        p_emb = self.p(x[:,1])
        c_emb = self.c(x[:,2])
        embs = u_emb, p_emb, c_emb
        x = torch.cat(embs, dim=1)
        out = self.layers(x).squeeze(1)
        return self.sig(out) * 1.05

class DataLoader():
    """ takes list of UserReactions and produces an iterator over batches """

    # vocabularize reaction_ids (get regression values off the reactions)
    reaction2value = {1: 1, 2: 0}

    def __init__(self,ur,bs,user2idx,post2idx,channel2idx):
        """
        Args:
        ur (list:UserReaction): list of user reactions with tg_channel_id
        bs (int): batch size
        user2idx (dict): vocab translating user_id from OlegDB to user embedding idx
        post2idx (dict): vocab translating post_id from OlegDB to post embedding idx 
        channel2idx (dict): vocab translating tg_channel_id from OlegDB to channel embedding idx 
        """

        self.bs = bs
        self.ur = ur

        self.cur = 0    # points on current idx in ur

        self.user2idx = user2idx
        self.post2idx = post2idx
        self.channel2idx = channel2idx

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
                channel_voc = self.channel2idx[self.ur[self.cur].tg_channel_id] 
                reac_voc = self.reaction2value[self.ur[self.cur].reaction_id]
                x_stack.append(tensor([int(user_voc), int(post_voc), int(channel_voc)]))
                y_stack.append(tensor(reac_voc, dtype=torch.float32))
            else:
                break
            self.cur += 1

        x_stack = torch.stack(x_stack)
        y_stack = torch.stack(y_stack)

        return x_stack, y_stack

class InferDataLoader():
    """ data loader for inferring """

    def __init__(self,posts,user,user2idx,post2idx,channel2idx):
        self.posts = posts
        self.user = user

        self.user2idx = user2idx
        self.post2idx = post2idx
        self.channel2idx = channel2idx
        
    def get_data(self):
        x_stack = []
        for p in self.posts:
            # only append those posts which are present in vocab
            try:
                x_stack.append(
                    tensor([
                        self.user2idx[self.user.id],
                        self.post2idx[p.id],
                        self.channel2idx[p.tg_channel_id]
                    ])
                )
            except KeyError:
                pass
        x_stack = torch.stack(x_stack)

        return x_stack



class OlegNN():
    """
    neural net functions for OlegAI
    """

    lr = 0.3        # learning rate
    bs = 512        # batch size 
    incremental_epochs = 4      # epochs for incremental training on new reactions
    full_learn_epochs = 12      # epochs to train on new learning dataset

    def __init__(
        self,
        user_lpv_len,               # length of latent parameters vectors
        post_lpv_len,
        channel_lpv_len,
        n_hidden,                   # hidden layer neurons qty
        nn_new_reactions_threshold, # how many new reactions to start incremental learning cycle
        nn_learning_timeout,        # min interval between learning cycles, seconds
        nn_full_learn_threshold,    # how many incremental learning cycles to start a full learning cycle
    ):
    
        self.user_lpv_len = int(user_lpv_len)
        self.post_lpv_len = int(post_lpv_len)
        self.channel_lpv_len = int(channel_lpv_len)

        self.n_hidden = int(n_hidden)

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
        
        # make vocabs
        idx2post = self.app.dba.get_post_ids()
        idx2user = self.app.dba.get_user_ids()
        idx2channel = self.app.dba.get_tg_channel_ids()

        self.model = UserPostChannelNN(
            (len(idx2user), self.user_lpv_len),
            (len(idx2post), self.post_lpv_len),
            (len(idx2channel), self.channel_lpv_len),
            idx2user,
            idx2post,
            idx2channel,
            self.n_hidden)

        #self.model = DotProduct(
        #    (len(idx2user),self.lpv_len),
        #    (len(idx2user),1),
        #    (len(idx2post),self.lpv_len),
        #    (len(idx2post),1),
        #    idx2user,
        #    idx2post
        #)

    def closest(self,posts,user):
        """ returns post closest to user """

        dl = InferDataLoader(posts, user, self.model.user2idx, self.model.post2idx, self.model.channel2idx)
        x = dl.get_data()
        preds = self.model.forward(x)
        dl = None

        # add some shuffle to result
        preds = self.shuffle(preds,amount=0.1)
        self.inf_preds = preds
        topidx = torch.argmax(preds).item()
        v_post_id = x[topidx][1]

        for post in posts:
            if post.id == self.model.idx2post[v_post_id]:
                posts = None
                return post

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

        # get reactions
        if full:
            ur = self.app.dba.get_user_reactions(with_channels=True)
            self.init_model()
        else:
            ur = self.app.dba.get_user_reactions(learned=0,with_channels=True)
            # add 96% old reactions to dataset or 4 batches
            ur = ur + self.app.dba.get_user_reactions(learned=1,limit=max(len(ur)*30,self.bs*4-len(ur)),with_channels=True)
        
        if not ur:
            self.learning = False
            return
        
        # DataLoader should provide batches like x.shape = (bs,3) y.shape = (bs,1)
        dl = DataLoader(ur, self.bs, self.model.user2idx, self.model.post2idx, self.model.channel2idx)

        if full:
            epochs = self.full_learn_epochs
        else:
            # prelearned embeddings already are in memory
            epochs = self.incremental_epochs

        self._learn(dl, self.model, epochs)

        # set learned=1 for reactions in dl
        self.app.dba.set_reactions_learned(ur)

        ur = None
        dl = None

        self.learning = False

    def _learn(self,dl,model,epochs=1):

        loss_func = nn.MSELoss()
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0)

        self.losses = []
        for e in range(epochs):
            for d in dl:
                optim.zero_grad()
                preds = model.forward(d[0])
                loss = loss_func(preds,d[1])
                self.losses.append(loss.item())

                loss.backward()
                optim.step()
        
        avg_loss = sum(self.losses)/len(self.losses)
        return avg_loss

    def _validate(self, dl, model, epochs=1):
        
        loss_func = nn.MSELoss()

        losses = []
        for e in range(epochs):
            for d in dl:
                preds = model.forward(d[0])
                loss = loss_func(preds,d[1])
                losses.append(loss.item())

        avg_loss = sum(losses)/len(losses)
        return avg_loss

    def learn_validate(self):

        self.init_model()

        ur = self.app.dba.get_user_reactions(with_channels=True, order='ASC')
        ur_l = ur[:int(len(ur)*0.8)] # learning ds
        ur_v = ur[int(len(ur)*0.8):] # validation ds
        random.shuffle(ur_l)
        random.shuffle(ur_v)

        dl_l = DataLoader(ur_l, self.bs, self.model.user2idx, self.model.post2idx, self.model.channel2idx)
        dl_v = DataLoader(ur_v, self.bs, self.model.user2idx, self.model.post2idx, self.model.channel2idx)
        
        self.l_epoch_losses = []
        self.v_epoch_losses = []
        print('Epoch | l_loss | v_loss')
        for e in range(self.full_learn_epochs):
            l_loss = self._learn(dl_l, self.model)
            self.l_epoch_losses.append(l_loss)
            v_loss = self._validate(dl_v, self.model)
            self.v_epoch_losses.append(v_loss)
            print(f'{e:5} | {round(l_loss,3):6} | {round(v_loss,3):6}')

    def shuffle(self, t:torch.Tensor, amount=0.05):
        """ adds shuffle to input tensor """
        return t + amount * torch.randn(t.shape)

    def add_emb(self, where, obj, emb):
        """ Adds new embedding to embedding matrix """

        # block until learning cycle done
        while self.learning:
            time.sleep(0.1)

        self.learning = True

        emb = emb.unsqueeze(0)

        voc = {
            'post': 'p',
            'user': 'u',
            'channel': 'c'
        }
        if where not in voc:
            raise Exception('[ OlegNN.add_emb ]: where must be "post", "user" or "channel"')

        embname = voc[where]

        fwd_voc = f'idx2{where}'
        bkwd_voc = f'{where}2idx'

        # add emb to embedding matrix
        if obj.id not in getattr(self.model, bkwd_voc):
            emb_rows = getattr(self.model, embname).weight.shape[0]

            # add row to emb
            setattr(
                self.model, embname,
                nn.Embedding.from_pretrained(
                    torch.cat(
                        (getattr(self.model, embname).weight.detach(), emb),
                        dim=0
                    )
                )
            )
            getattr(self.model, embname).weight.requires_grad_(True)

            # add value to vocab
            getattr(self.model, fwd_voc).append(obj.id)
            getattr(self.model, bkwd_voc)[obj.id] = emb_rows

        self.learning = False

    def handle_new_obj(self, where, obj):
        """ Adds embedding to model for new object """

        voc = ['user', 'post', 'channel']
        if where not in voc:
            raise Exception(f'[ OlegNN.handle_new_obj ]: where attribute incorrect: {where}')
        
        embname = f'{where}_lpv_len'
        emb = torch.randn(getattr(self, embname))
        self.add_emb(where, obj, emb)