import torch
import warnings

warnings.filterwarnings('ignore')
from muon import prot as pt
import torch
from tqdm.auto import tqdm
torch.set_default_tensor_type(torch.DoubleTensor)
from sklearn import metrics
from torch.utils.data import (DataLoader, Dataset)
import pandas as pd
import scipy
import time

from scmGPT.utils import *
from scmGPT.model import GPT
from scmGPT.train_model import Trainer
same_seeds(2023)
gap = 128
model_name = "scmGPT"
data_name = "cite"
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

pt.pp.clr(adt_adata, axis=1)
sc.pp.log1p(adt_adata)

###train data
train_modX = adata[~adata.obs[obs_values].isin(donnor_list)]
train_X = train_modX.obsm['X_pca']  #

train_modY = adt_adata[~adt_adata.obs[obs_values].isin(donnor_list)]
train_Y = train_modY.X  # obsm['X_pca']  #
train_Y = tomatrix2(train_Y)

test_modX = adata[adata.obs[obs_values].isin(donnor_list)]
test_X = test_modX.obsm['X_pca']  ##.X

test_modY = adt_adata[adt_adata.obs[obs_values].isin(donnor_list)]
test_Y = test_modY.X
test_Y = tomatrix2(test_Y)

del adata, adt_adata, train_modX, train_modY  # ,test_modX,test_modY

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