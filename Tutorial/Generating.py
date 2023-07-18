import torch
import warnings

warnings.filterwarnings('ignore')
import scanpy as sc
import os
from muon import prot as pt

torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from torch.utils.data import Dataset
import math
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from tqdm.auto import tqdm
from sklearn.metrics import *

torch.set_default_tensor_type(torch.DoubleTensor)
import geomloss
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
import scanpy as sc
from torch.utils.data import (DataLoader, Dataset)
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


def label_transfer(ref, query, label):
    from sklearn.neighbors import KNeighborsClassifier

    X_train = ref
    y_train = label
    X_test = query

    knn = knn = KNeighborsClassifier().fit(X_train, y_train)
    y_test = knn.predict(X_test)

    return y_test


def run_SVM(x_train, y_train, x_test, kernel="rbf", seed=2021):
    '''Fit SVM model with a RBF kernel
    SVM decision_function_shape: can be one-vs-one or one-vs-rest, in our case,
    according to our input, we should use one-vs-rest
    '''
    if "rbf" == kernel:
        from sklearn.svm import SVC
        model = SVC(decision_function_shape="ovr", kernel=kernel, random_state=seed)
        # model = SVC(decision_function_shape="ovo", kernel=kernel, random_state=seed)
    elif "linear" == kernel:
        from sklearn.svm import LinearSVC
        model = LinearSVC(multi_class='ovr', random_state=seed)

    ## fit model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2023)


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # print("B, T, C",B, T, C)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # print("q, k, v",q, k, v)
        q = F.relu(q)
        k = F.relu(k)
        v = F.relu(v)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        att = F.relu(att)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 2 * config.n_embd),
            c_proj=nn.Linear(2 * config.n_embd, config.n_embd),
            act=nn.ReLU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class scDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data)
        label = torch.from_numpy(self.label)

        x = torch.tensor(data[idx])
        y = torch.tensor(label[idx])
        return x, y


class GPT(nn.Module):

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.entreg = .1
        C.p = 2
        C.h = 2
        C.loss1 = 50
        C.mod2_dim = 134
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.conf = config
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                                       # names follow the huggingface naming conventions
                                       # GPT-1
                                       'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                                       # GPT-2 configs
                                       'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                                       'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                                       'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                                       'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                                       # Gophers
                                       'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512),
                                       # (there are a number more...)
                                       # I made these tiny models up
                                       'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
                                       'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
                                       'gpt-nano': dict(n_layer=1, n_head=config.h, n_embd=config.n_embd),
                                   }[config.model_type])
        print("config.vocab_size, config.n_embd", config.vocab_size, config.n_embd)
        self.pro = nn.Linear(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))  # config.n_embd*config.block_size
        # self.lm_head = nn.Linear(config.n_embd, config.mod2_dim, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.mod2_dim)  # , bias=False
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def cross_mod(self, mod1, mod2=None):
        idx = torch.tensor(mod1, dtype=torch.double)
        device = idx.device
        b, t, v = idx.size()

        num_cls = int(self.conf.block_size / 2)

        if (self.conf.block_size % 2 == 0):
            cls1 = torch.zeros(num_cls).long().to(device)
        else:
            cls1 = torch.zeros(num_cls + 1).long().to(device)

        cls2 = torch.ones(num_cls).long().to(device)
        cls = torch.cat((cls1, cls2), dim=0).to(device)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        cls_emb = self.transformer.wpe(cls)  # + cls_emb

        x = self.transformer.drop(idx + pos_emb + cls_emb)
        for block in self.transformer.h:
            x = F.relu(x)
            x = block(x)
        x = F.relu(x)
        x = self.transformer.ln_f(x)
        x = F.relu(x)
        x = torch.mean(x, dim=1)
        emb = x
        logits = self.lm_head(x)
        mod_logits = F.relu(logits)

        # if we are given some desired targets also calculate the loss
        loss = None
        # criterion = nn.CrossEntropyLoss()

        if mod2 is not None:
            targets = torch.tensor(mod2, dtype=torch.double)
            loss1 = F.mse_loss(mod_logits, targets) ** 0.5

            #             dim2 = int(mod_logits.shape[1] / 2)
            #             loss2 = F.mse_loss(mod_logits[:,dim2:], targets[:,dim2:]) ** 0.5

            #             print("mod_logits.shape",mod_logits.shape)
            #             print("dim2",dim2)

            loss = loss1  # + loss2#*10 #*10 # loss1 #+ loss2  loss1 +
        return loss, emb, mod_logits

    def forward(self, X, Y):

        loss1, emb_mod, mod1_logits2 = self.cross_mod(X, Y)

        emb_mod1 = emb_mod[:, :int(emb_mod.shape[1] / 2)]
        emb_mod2 = emb_mod[:, int(emb_mod.shape[1] / 2):]
        #         loss3 = F.mse_loss(emb_mod1, emb_mod2) ** 0.5
        loss3 = torch.tensor(0)
        loss = self.conf.loss1 * loss1 + loss3  # / 10
        return emb_mod1, emb_mod2, loss, loss1, loss3, mod1_logits2

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate,
                                      betas=train_config.betas)  # ,weight_decay=train_config.weight_decay
        return optimizer


from typing import Tuple
import scipy


def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return foscttm_x, foscttm_y


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.epoch = 100
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.95, 0.99)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True,
                                  pin_memory=True)

        model.train()
        n_epochs = config.epoch
        for epoch in range(n_epochs):

            train_loss = []
            train_loss1 = []
            train_loss3 = []
            emb_mod1s = []
            emb_mod2s = []
            mod1_logits2s = []
            mod2_logits1s = []
            for batch in tqdm(train_loader):
                X, Y, = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                emb_mod1, emb_mod2, self.loss, self.loss1, self.loss3, mod1_logits2 = model(X, Y)
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                train_loss.append(self.loss.item())
                train_loss1.append(self.loss1.item())
                train_loss3.append(self.loss3.item())

                if (self.device == 'cuda'):
                    emb_mod1s.extend(emb_mod1.cpu().detach().numpy())  # .numpy()
                    emb_mod2s.extend(emb_mod2.cpu().detach().numpy())  # .numpy()
                    mod1_logits2s.extend(mod1_logits2.cpu().detach().numpy())  # .numpy()
                else:
                    emb_mod1s.extend(emb_mod1.detach().numpy())  # .numpy()
                    emb_mod2s.extend(emb_mod2.detach().numpy())  # .numpy()
                    mod1_logits2s.extend(mod1_logits2.detach().numpy())  # .numpy()

            train_loss = sum(train_loss) / len(train_loss)
            train_loss1 = sum(train_loss1) / len(train_loss1)
            train_loss3 = sum(train_loss3) / len(train_loss3)

            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f},"
                  f"loss1 = {train_loss1:.5f},"
                  f"loss3 = {train_loss3:.5f},")

        emb_mod1s = np.asarray(emb_mod1s)
        emb_mod2s = np.asarray(emb_mod2s)

        mod1_logits2s = np.asarray(mod1_logits2s)

        return emb_mod1s, emb_mod2s, mod1_logits2s  # , mod2_logits1s


def tomatrix(train_mod1_X):
    if not isinstance(train_mod1_X.X, np.ndarray):
        X = np.asarray(train_mod1_X.X.todense())
    else:
        X = train_mod1_X.X

    return X


def tomatrix2(train_mod1_X):
    if not isinstance(train_mod1_X, np.ndarray):
        X = np.asarray(train_mod1_X.todense())
    else:
        X = train_mod1_X

    return X


# gap, data_path, data_name,topgenes
def GeneEmbeding(X, gap):
    # gap = 10240
    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        for k in range(0, length, gap):
            if (k + gap > length):
                a = single_cell[length - gap:length]
            else:
                a = single_cell[k:k + gap]
            feature.append(a)
        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)
    print("single_cell_list.shape", single_cell_list.shape)
    return single_cell_list


def getXY(mod_paths, mod_names, gap):
    adata_mod1 = sc.read_h5ad(mod_paths[0] + mod_names[0])
    adata_mod1.var_names_make_unique()
    adata_mod1.obs['domain_id'] = 0
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1, n_top_genes=2000)
    adata_mod1 = adata_mod1[:, adata_mod1.var['highly_variable']]

    print("adata_mod1.shape", adata_mod1.shape)

    adata_mod2 = sc.read_h5ad(mod_paths[1] + mod_names[1])
    adata_mod2.var_names_make_unique()
    adata_mod2.obs['domain_id'] = 1
    sc.pp.normalize_total(adata_mod2)
    sc.pp.log1p(adata_mod2)

    print("adata_mod2.shape", adata_mod2.shape)

    X1 = adata_mod1.X  # obsm['X_pca']#.todense()
    X2 = adata_mod2.X  # .todense()
    if not isinstance(X1, np.ndarray):
        X1 = X1.todense()

    if not isinstance(X2, np.ndarray):
        X2 = X2.todense()

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X1 = np.concatenate((X1, X2), axis=1)
    X1_ = np.concatenate((X2, X1,), axis=1)
    train_labels = adata_mod1.obs["cell_type"].values
    # test_labels = adata_mod2.obs["cell_type"].values
    test_labels = adata_mod2.obs["cell_type"].values

    Y1 = X1

    X1 = GeneEmbeding(X1, gap)

    return X1, train_labels, Y1, test_labels, adata_mod1, adata_mod2  # single_cell_list, labelss, cell_types


def getXY2(mod_paths, mod_names, gap):
    if (True):
        # step1: 获得两种模态数据并且进行预处理
        adata_mod1 = sc.read_h5ad(mod_paths[0] + mod_names[0])
        adata_mod1.var_names_make_unique()
        adata_mod1.obs['domain_id'] = 0
        sc.pp.normalize_total(adata_mod1)
        sc.pp.log1p(adata_mod1)
        sc.pp.highly_variable_genes(adata_mod1, n_top_genes=2000)
        adata_mod1 = adata_mod1[:, adata_mod1.var['highly_variable']]

        print("adata_mod1.shape", adata_mod1.shape)

        adata_mod2 = sc.read_h5ad(mod_paths[1] + mod_names[1])
        adata_mod2.var_names_make_unique()
        adata_mod2.obs['domain_id'] = 1
        sc.pp.normalize_total(adata_mod2)
        sc.pp.log1p(adata_mod2)

        print("adata_mod2.shape", adata_mod2.shape)

        X1 = adata_mod1.X  # obsm['X_pca']#.todense()
        X2 = adata_mod2.X  # .todense()
        if not isinstance(X1, np.ndarray):
            X1 = X1.todense()

        if not isinstance(X2, np.ndarray):
            X2 = X2.todense()

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X1 = np.concatenate((X1, X2), axis=1)
        X1_ = np.concatenate((X2, X1,), axis=1)
        train_labels = adata_mod1.obs["cell_type"].values
        # test_labels = adata_mod2.obs["cell_type"].values
        test_labels = adata_mod2.obs["cell_type"].values

        Y1 = X1

        X1 = GeneEmbeding(X1, gap)

    return X1, train_labels, Y1, test_labels, adata_mod1, adata_mod2  # single_cell_list, labelss, cell_types


import time

gap = 128
model_name = "scmGPT"
data_name = "newcite"
import os

start = time.time()

log_dir = "log/" + model_name + "/" + data_name + "/"
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)

data_path = "../../Dataset/ADT_GEX/" + data_name + "/"

donnor_list = ['donor1', 'donor2', 'donor3']  # ,'donor4', 'donor5', 'donor6'
obs_values = 'DonorNumber'

adata = sc.read_h5ad(data_path + data_name + "_rna.h5ad")
adt_adata = sc.read_h5ad(data_path + data_name + "_adt.h5ad")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, )
adata = adata[:, adata.var['highly_variable']]
sc.tl.pca(adata, n_comps=adt_adata.shape[1], svd_solver="auto")

# sc.pp.normalize_total(adt_adata)
# sc.pp.log1p(adt_adata)
# pt.pp.clr(adt_adata,axis=1)
pt.pp.clr(adt_adata, axis=1)
sc.pp.log1p(adt_adata)

###train data
train_modX = adata[~adata.obs[obs_values].isin(donnor_list)]
train_X = train_modX.obsm['X_pca']  #
# train_X = tomatrix2(train_X)

train_modY = adt_adata[~adt_adata.obs[obs_values].isin(donnor_list)]
train_Y = train_modY.X  # obsm['X_pca']  #
train_Y = tomatrix2(train_Y)

test_modX = adata[adata.obs[obs_values].isin(donnor_list)]
test_X = test_modX.obsm['X_pca']  ##.X
# test_X = tomatrix2(test_X)

test_modY = adt_adata[adt_adata.obs[obs_values].isin(donnor_list)]
test_Y = test_modY.X
test_Y = tomatrix2(test_Y)

del adata, adt_adata, train_modX, train_modY  # ,test_modX,test_modY

print("train_modX.shape", train_X.shape)
print("train_modY.shape", train_Y.shape)
print("test_mod1.shape", test_X.shape)
print("test_mod2.shape", test_Y.shape)

Y1 = np.concatenate((train_X, train_Y), axis=1)
print("Y1.shape", Y1.shape)

train_X = GeneEmbeding(train_X, gap)

impute_train_Y = np.zeros_like(train_Y)
impute_train_Y = GeneEmbeding(impute_train_Y, gap)

X1 = np.concatenate((train_X, impute_train_Y), axis=1)

a, b, c = X1.shape
print("a,b,c,", a, b, c)
train_dataset = scDataSet(data=X1, label=Y1)

del train_X, train_Y, impute_train_Y  # X1,Y1,

model_name = "end4"

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = c
model_config.block_size = b
model_config.n_embd = c
model_config.embd_pdrop = 0.1
model_config.resid_pdrop = 0.1
model_config.attn_pdrop = 0.1
model_config.loss1 = 50
model_config.h = 16
model_config.mod2_dim = Y1.shape[1]
print("model_config.mod2_dim ", model_config.mod2_dim)
print("model_config.vocab_size", model_config.vocab_size)
model = GPT(model_config)

train_config = Trainer.get_default_config()
train_config.epoch = 10
train_config.learning_rate = 1e-3
train_config.batch_size = 1024  # 10240
trainer = Trainer(train_config, model, train_dataset)

log_dir = "log/" + str(model_name) + "/" + data_name + "/"
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)

emb_mod1s, emb_mod2s, mod1_logits2s, = trainer.run()  # mod2_logits1s
print("emb_mod1s.shape", emb_mod1s.shape)
print("emb_mod2s.shape", emb_mod2s.shape)

model.eval()
gene_num = test_X.shape[1]
adt_num = test_Y.shape[1]

test_Y1 = np.concatenate((test_X, test_Y), axis=1)
print("test_Y1.shape", test_Y1.shape)

impute_test_Y = np.zeros_like(test_Y)
test_X = GeneEmbeding(test_X, gap)
impute_test_Y = GeneEmbeding(impute_test_Y, gap)
test_X1 = np.concatenate((test_X, impute_test_Y), axis=1)

test_dataset = scDataSet(data=test_X1, label=test_Y1)
batchsize = len(test_X1)

del test_X1, test_Y1, impute_test_Y, test_X  # ,test_Y
# del X1,Y1
import time

test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, pin_memory=True)

test_mse = []
test_rna_mse = []
test_adt_mse = []

test_loss = []
all_logits = []
all_y = []
all_pred_adt = []
all_y_adt = []

device = 'cuda'
# gene_num = 210
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        emb_mod1, emb_mod2, loss, loss1, loss3, logits = model(x, y)
    print("logits.shape", logits.shape, "y.shape", y.shape)

    if (device == 'cuda'):
        logits = logits.cpu().detach().numpy()
        y = y.detach().cpu().numpy()
    else:
        logits = logits.detach().numpy()
        y = y.detach().numpy()

    rna_pred = logits[:, :gene_num]
    adt_pred = logits[:, gene_num:]

    y_rna = y[:, :gene_num]
    y_adt = y[:, gene_num:]

    RMSE = metrics.mean_squared_error(logits, y) ** 0.5
    rna_RMSE = metrics.mean_squared_error(rna_pred, y_rna) ** 0.5
    adt_RMSE = metrics.mean_squared_error(adt_pred, y_adt) ** 0.5

    all_y_adt.extend(y_adt)
    all_y.extend(y)
    all_pred_adt.extend(adt_pred)
    all_logits.extend(logits)
    test_mse.append(RMSE)
    test_rna_mse.append(rna_RMSE)
    test_adt_mse.append(adt_RMSE)

    test_loss.append(loss.item())

test_mse = sum(test_mse) / len(test_mse)
test_rna_mse = sum(test_rna_mse) / len(test_rna_mse)
test_adt_mse = sum(test_adt_mse) / len(test_adt_mse)
test_loss = sum(test_loss) / len(test_loss)
print("---------------------------------------------end test---------------------------------------------")
print("test_loss:", test_loss, "test_mse:", test_mse, "test_rna_mse:", test_rna_mse, "test_adt_mse:", test_adt_mse, )

end = time.time()
all_y_adt = np.asarray(all_y_adt)
all_pred_adt = np.asarray(all_pred_adt)
pd.DataFrame(all_pred_adt, index=test_modY.obs_names).to_csv(log_dir + "adt_pred.csv", header=False)

def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr
res = []

true_rna = all_y_adt
X_hat = all_pred_adt
pr, sr = comp_cor_flatten(X_hat.flatten(), true_rna.flatten())
mse = np.mean((true_rna.flatten()-X_hat.flatten())**2)
res.append(['ADT', 'ADT', pr, sr, mse])

gene_names = test_modY.var_names.values
_df = pd.DataFrame({
    'RNA':gene_names,
    'Pearson r':[np.corrcoef(X_hat[:,i], true_rna[:,i])[0,1] for i in np.arange(len(gene_names))],
    'Spearman r':[scipy.stats.spearmanr(X_hat[:,i], true_rna[:,i])[0] for i in np.arange(len(gene_names))],
    'MSE':np.mean((X_hat-true_rna)**2, axis=0),
    'RMSE':np.sqrt(np.mean((X_hat-true_rna)**2, axis=0)),
})
print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
_df.to_csv(log_dir+'res_scmGPT_adt.csv')


all_time = end - start
with open(log_dir + "end_norm.txt", "a") as f:
    f.writelines("---------------------------log_dir:" + log_dir + "-------------------------" + "\n")
    f.writelines("model_config:" + "\n" + str(model_config) + "\n")
    f.writelines('train_config:' + "\n" + str(train_config) + "\n")
    f.writelines("test_loss:" + str(test_loss) + "\n")
    f.writelines('test_mse:' + str(test_mse) + "\n")
    f.writelines("all_time:" + str(all_time) + "\n")
    f.writelines('test_adt_mse:' + str(test_adt_mse) + "\n")
    f.writelines('pr:' + str(pr) + "\n")
    f.writelines('sr:' + str(sr) + "\n")