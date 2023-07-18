import numpy as np
import scanpy as sc

def GeneEmbeding(X, gap):
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
    return single_cell_list

def getXY_scRNA_scATAC(mod_paths, mod_names, gap):
    # step1: 获得两种模态数据并且进行预处理
    adata_mod1 = sc.read_h5ad(mod_paths[0] + mod_names[0])
    adata_mod1.var_names_make_unique()
    adata_mod1.obs['domain_id'] = 0
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)

    adata_mod2 = sc.read_h5ad(mod_paths[1] + mod_names[1])
    adata_mod2.var_names_make_unique()
    adata_mod2.obs['domain_id'] = 1
    sc.pp.normalize_total(adata_mod2)
    sc.pp.log1p(adata_mod2)


    data_cm = adata_mod1.concatenate(adata_mod2)

    sc.pp.highly_variable_genes(data_cm, n_top_genes=2000)

    data_cm = data_cm[:, data_cm.var['highly_variable']]
    #sc.pp.scale(data_cm)
    import uniport as up
    adata_mod1 = data_cm[:len(adata_mod1)]
    adata_mod2 = data_cm[len(adata_mod1):]
    # sc.tl.pca(adata_mod1, n_comps=100, svd_solver="auto")
    # sc.tl.pca(adata_mod2, n_comps=100, svd_solver="auto")

    X1 = adata_mod1.X#obsm['X_pca']#.todense()
    X2 = adata_mod2.X#obsm['X_pca']#.todense()
    if not isinstance(X1, np.ndarray):
        X1 = X1.todense()

    if not isinstance(X2, np.ndarray):
        X2 = X2.todense()

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.concatenate((X1,X2),axis=1)

    X1 = GeneEmbeding(X1, gap)
    X2 = GeneEmbeding(X2, gap)

    X1 = np.concatenate((X1,X2),axis=1)


    return X1, Y1,adata_mod1,adata_mod2  # single_cell_list, labelss, cell_types


def getXY_protein(mod_paths, mod_names):
    # step1: 获得两种模态数据并且进行预处理
    adata_mod1 = sc.read_h5ad(mod_paths[0] + mod_names[0])
    adata_mod1.var_names_make_unique()
    adata_mod1.obs['domain_id'] = 0
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1, n_top_genes=2000)
    adata_mod1 = adata_mod1[:, adata_mod1.var['highly_variable']]

    adata_mod2 = sc.read_h5ad(mod_paths[1] + mod_names[1])
    adata_mod2.var_names_make_unique()
    adata_mod2.obs['domain_id'] = 1

    num_cell1 = adata_mod1.shape[1]
    num_cell2 = adata_mod2.shape[1]

    features = min(num_cell1, num_cell2)

    if (features < 40):
        comp = 30
        gap = 16
        h = 4
    elif (features < 50):
        comp = 40
        gap = 16
        h = 4
    elif (50 < features < 100):
        comp = 50
        gap = 32
        h = 8
    elif (100 < features < 200):
        comp = 100
        gap = 64
        h = 16
    elif (200 < features < 300):
        comp = 200
        gap = 192
        h = 32
    elif (300 < features):
        comp = 200
        gap = 192
        h = 32
    else:
        comp = int(features/2) * 2
        gap = int(features/2)
        h = 2

    sc.tl.pca(adata_mod1, n_comps=comp, svd_solver="auto")
    sc.tl.pca(adata_mod2, n_comps=comp, svd_solver="auto")

    X1 = adata_mod1.obsm['X_pca']
    X2 = adata_mod2.obsm['X_pca']

    if not isinstance(X1, np.ndarray):
        X1 = X1.todense()

    if not isinstance(X2, np.ndarray):
        X2 = X2.todense()

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.concatenate((X1,X2),axis=1)

    X1 = GeneEmbeding(X1, gap)
    X2 = GeneEmbeding(X2, gap)

    X1 = np.concatenate((X1,X2),axis=1)

    return X1, Y1, adata_mod1, adata_mod2, h  # single_cell_list, labelss, cell_types
