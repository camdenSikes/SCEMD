import grakel.kernels.weisfeiler_lehman_optimal_assignment
from grakel.datasets import fetch_dataset
from grakel.kernels import LovaszTheta, GraphletSampling, WeisfeilerLehmanOptimalAssignment, WeisfeilerLehman
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import numpy as np
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='COLLAB',
                        help='Provide the dataset name (available in TUDataset repository)')
    parser.add_argument('-k', '--kernel', type=str, default='Graphlet', choices=['Graphlet', 'Lovasz', 'Weisfeiler', 'OptimalAssignment'])
    args = parser.parse_args()
    dataset = fetch_dataset(args.dataset, verbose=False)
    G = dataset.data
    graphlabels = dataset.target
    # inds = random.sample(range(len(G)), 800)
    # G = [G[i] for i in inds]
    # graphlabels = [graphlabels[i] for i in inds]

    model = GraphletSampling(sampling={"n_samples": 50})
    if args.kernel == 'OptimalAssignment':
        model = WeisfeilerLehmanOptimalAssignment()
    elif args.kernel == 'Lovasz':
        model = LovaszTheta()
    elif args.kernel == 'Weisfeiler':
        model = WeisfeilerLehman()


    tic = time.perf_counter()
    k = model.fit_transform(G)
    toc = time.perf_counter()
    print('Time taken to compute kernel matrix: ', toc - tic)

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

if __name__ == '__main__':
    main()


