# SCEMD
Spectral Clustering Earth Mover's Distance Graph Kernel

## Dependencies
This code is written in Python 3.11 and has the following dependencies:
- `numpy==1.26.4`
- `scipy==1.13.0`
- `scikit-learn==1.4.0`
- `POT==0.9.3` ([github](https://github.com/rflamary/POT))
- `Cython==3.0.9`

## Datasets

Download the datasets from here: https://chrsmrrs.github.io/datasets/docs/datasets/ and then extract them into the "Datasets" folder

## Running the code

The following runs the SCEMD graph kernel on the MUTAG dataset using 8 eigenvectors and 8 slices per dimension:

`python main.py -d MUTAG -c -e 8 -k 8`

Run `python main.py -h` to view settings options. For the standard EMD kernel, do not include the -c option.

```bash
usage: main.py [-h] [-d DATASET] [-l] [-c] [-e NUMEIGS] [-k NUMSLICES]

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Provide the dataset name (name of TUDataset-formatted directory in Datasets directory)
  -l, --labeled         Enable use of node labels
  -c, --clustering      Enable clustering of node embeddings
  -e NUMEIGS, --numeigs NUMEIGS
                        Number of eigenvectors to use for embedding
  -k NUMSLICES, --numslices NUMSLICES
                        Number of slices per dimension, only used if clustering is enabled
```
## Comparisons
To compare against the graph kernels implemented in GraKel, first install grakel and then you can run:
`python grakeltests.py -d COLLAB -k Graphlet`

## Cite
tbd



