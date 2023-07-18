# scmGPT
![image]()

># Introduction

scmGPT(single-cell multimodal generative pre-training transformer), a Transformer-based model, can be used to integrate and generate single-cell omics data. 

Instructions and examples are provided in the following tutorials.

># Requirement
```
Python 3.9.12
PyTorch >= 1.5.0
numpy
pandas
scipy
sklearn
Scanpy
random
```
## Input file
```
the first  modality(scRNA-seq)  dataset.
the second modality(scATAC-seq)  dataset.
```

## Output file
```
After the scmGPT model, the model will be save at: "log/scmGPT.tar".
The latent representations for each modality  are saved in the log/mod1.npy,log/mod2.npy
```

[//]: # (```)

## Usage

### 
```Python
import scmGPT as sg
mod1,mod2 = sg.scmGPT(s, referece_datapaths, Train_names, Testdata_path,Testdata_name)
```

in which 

- **s=The length of sub-vector**,
- **referece_datapaths=The path of annotated scRNA-seq datasets**
- **Train_names=The name of annotated scRNA-seq datasets** 
- **Testdata_path=The path of query scRNA-seq datasets**
- **Testdata_name=The name of query scRNA-seq datasets** 


># Tutorial

## processsed Data

## The scRNA-seq datasets pre-processing code

># Paper Link

