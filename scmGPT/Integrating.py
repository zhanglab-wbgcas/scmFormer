import torch
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import os

from scmGPT.utils import *
from scmGPT.model import GPT
from scmGPT.train_model import Trainer
same_seeds(2023)

import time
data_name = "10x-Multiome-Pbmc10k"
test_GEX = "C:/D/code/pycharmCode/04Mycode/multiModel/Dataset/scglue/"+ data_name +"/"
test_ATAC = "C:/D/code/pycharmCode/04Mycode/multiModel/Dataset/scglue/"+ data_name +"/"

test_paths = [test_GEX, test_ATAC]
test_names = [data_name + "-RNA.h5ad",data_name + "-ATAC_gene.h5ad",]

gap = 128

X1, Y1,adata_mod1,adata_mod2 = getXY_scRNA_scATAC(test_paths, test_names, gap)
start = time.time()
a, b, c = X1.shape
train_dataset = scDataSet(data = X1, label = Y1)
model_name = "Integrating_scRNA_scATAC"

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = c
model_config.block_size = b
model_config.n_embd = c
model_config.embd_pdrop = 0.0
model_config.resid_pdrop = 0.0
model_config.attn_pdrop = 0.0
model_config.loss1 = 50
model_config.h = 8
model_config.mod2_dim = Y1.shape[1]
print("model_config.mod2_dim ", model_config.mod2_dim)
print("model_config.vocab_size", model_config.vocab_size)
model = GPT(model_config)

train_config = Trainer.get_default_config()
train_config.epoch = 1
train_config.learning_rate = 1e-4
train_config.batch_size = 32  # 10240
trainer = Trainer(train_config, model, train_dataset)

log_dir = "log/" + str(model_name) + "/"+ str(train_config.epoch) + "/" + data_name + "/"
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)

emb_mod1s, emb_mod2s,mod1_logits2s, = trainer.run() #mod2_logits1s
np.save(log_dir+"emb_mod1s.npy",emb_mod1s)
np.save(log_dir+"emb_mod2s.npy",emb_mod2s)
end = time.time()
all_time = end - start

