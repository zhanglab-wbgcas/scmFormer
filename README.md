# scmFormer
![image]()

># Introduction

scmFormer(single-cell multi-modal/multi-task transformer), a Transformer-based model, can be used to integrate and generate single-cell omics data. 

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
After the scmFormer model, the model will be save at: "log/scmFormer.tar".
The latent representations for each modality  are saved in the log/mod1.npy,log/mod2.npy
```

[//]: # (```)


># Tutorial

## processsed Data

## The scRNA-seq datasets pre-processing code

># Paper Link

