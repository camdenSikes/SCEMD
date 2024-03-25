import argparse
import time
import random
from scipy import sparse as sp
import numpy as np
from emdkernel import *
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MUTAG',
                        help='Provide the dataset name (name of TUDataset-formatted directory in Datasets directory)')
    parser.add_argument('-l', '--labeled', default=False, action='store_true',
                        help='Enable use of node labels')
    parser.add_argument('-c', '--clustering', default=False, action='store_true',
                        help='Enable clustering of node embeddings')
    parser.add_argument('-e', '--numeigs', type=int, default=6,
                        help='Number of eigenvectors to use for embedding')
    parser.add_argument('-k', '--numslices', type=int, default=6,
                        help='Number of slices per dimension, only used if clustering is enabled')
    args = parser.parse_args()
    adjmats, graphlabels, nodelabels = load_sparse_adjmats(args.dataset, args.labeled)
    # random subset for speeding up testing
    inds = random.sample(range(len(adjmats)), 800)
    adjmats = [adjmats[i] for i in inds]
    graphlabels = [graphlabels[i] for i in inds]
    if args.labeled:
        nodelabels = [nodelabels[i] for i in inds]
    tic = time.perf_counter()
    if args.clustering:
        k = scemdkernel(adjmats, d=args.numeigs, k=args.numslices, labeled=args.labeled, labels=nodelabels)
    else:
        k = emdkernel(adjmats, d=args.numeigs, labeled=args.labeled, labels=nodelabels)
    toc = time.perf_counter()
    print('Time taken to compute kernel matrix: ', toc - tic)
    #test if positive semidefinite
    try:
        np.linalg.cholesky(k)
    except:
        # if not positive semidefinite, use exponential kernel
        maxval = np.max(k)
        k = np.exp(-k/maxval)

    #test kernel
    accuracy_scores = []
    np.random.seed(42)
    y = np.array(graphlabels)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in cv.split(k, graphlabels):
        k_train = k[train_index][:, train_index]
        k_test = k[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]
        gs = SVC(C=100, kernel='precomputed').fit(k_train, y_train)
        y_pred = gs.predict(k_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
        np.mean(accuracy_scores) * 100,
        np.std(accuracy_scores) * 100))

    return


def load_sparse_adjmats(dataset, labeled):
    with open("Datasets/" + dataset + "/" + dataset + "_A.txt") as reader:
        edges = reader.readlines()
    with open("Datasets/" + dataset + "/" + dataset + "_graph_indicator.txt") as reader:
        graphind = [int(x) for x in reader.readlines()]
    with open("Datasets/" + dataset + "/" + dataset + "_graph_labels.txt") as reader:
        graphlab = [int(x) for x in reader.readlines()]
    nodelab = None
    if labeled:
        with open("Datasets/" + dataset + "/" + dataset + "_node_labels.txt") as reader:
            nodelab = [int(x) for x in reader.readlines()]
    N = len(graphlab)
    nodelists = []
    adjmats = []
    labels = []
    for i in range(1, N + 1):
        nodes = [j for j, x in enumerate(graphind) if x == i]
        nodelists.append(nodes)
        if labeled:
            label = [nodelab[j - 1] for j in nodes]
            label = np.array(label)
            labels.append(label)
        adjmat = sp.lil_array((len(nodes), len(nodes)))
        adjmats.append(adjmat)
    for edge in edges:
        pair = edge.split(',')
        node1 = int(pair[0]) - 1
        node2 = int(pair[1]) - 1
        i = graphind[node1] - 1
        node1id = nodelists[i].index(node1)
        node2id = nodelists[i].index(node2)
        adjmats[i][node1id, node2id] = 1
        adjmats[i][node2id, node1id] = 1
    for i in range(len(adjmats)):
        adjmats[i] = sp.csr_array(adjmats[i])
    return adjmats, graphlab, labels


if __name__ == '__main__':
    main()
