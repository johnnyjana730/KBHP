# KBHP

KBHP: Knowledge Based Hyperbolic Propagation, SIGIR 2021

This repository is the implementation of KBHP ([ACM](https://dl.acm.org/doi/10.1145/3404835.3462980)):
> Chang-You Tai, and Lun-Wei Ku. SIGIR 2021. KBHP: Knowledge Based Hyperbolic Propagation


## Introduction
We propose the knowledge basedhyperbolic propagation framework (KBHP), a KG-aware recommendation model which includes hyper-bolic components for calculating the importance of KG attributes’ relatives to achieve better knowledge propagation.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{10.1145/3404835.3462980,
author = {Tai, Chang-You and Huang, Chien-Kun and Huang, Liang-Ying and Ku, Lun-Wei},
title = {Knowledge Based Hyperbolic Propagation},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3462980},
doi = {10.1145/3404835.3462980},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1945–1949},
numpages = {5},
keywords = {hyperbolic embedding learning, knowledge graph, graph neural network, recommendation},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```



## Files in the folder

- `data/`: datasets
  - `MovieLens-1M/`
  - `amazon-book_20core/`
  - `last-fm_50core/`
  - `music/`
- `src/model/`: implementation of KBHP.
- `output/`: storing log files
- `misc/`: storing users being evaluating, popular items, and sharing embeddings.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* torch == 1.8.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0

## Build Environment(conda)
```
$ cd KBHP
$ conda deactivate
$ conda env create -f requirements.yml
$ conda activate KBHP
```

## Example to Run the Codes

* KBHP
```
$ cd bash
$ bash bash_run.sh $dataset $gpu
```

* other baseline models, pls refer to (https://github.com/johnnyjana730/MVIN)

* `dataset`
  * It specifies the dataset.
  * Here we provide three options, including  * `az`, `mv`, `la`, or `mu`.

* `gpu`
  * It specifies the gpu, e.g. * `0`, `1`, and `2`.

# Issue

* `main_run.sh syntax error near unexpected token elif`
```
$ sed -i -e 's/\r$//' *.sh
```
