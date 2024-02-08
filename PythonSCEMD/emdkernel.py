import numpy as np
import scipy.sparse.linalg as la
import ot


def label_dists(embed1, embed2, labs1, labs2):
    costs = ot.dist(embed1, embed2)
    max_dist = np.sqrt(len(embed1[0]))
    for i in range(len(labs1)):
        for j in range(len(labs2)):
            if labs1[i] != labs2[j]:
                costs[i][j] = max_dist
    return costs


def label_dists_coarse(embed1, embed2, labs1, labs2):
    costs = ot.dist(embed1, embed2)
    labcosts = ot.dist(labs1, labs2)
    return costs + labcosts


def emdkernel(adjmats, d=6, labeled=False, labels=None):
    n = len(adjmats)
    M = np.zeros((n, n))
    embeddings = []
    for adjmat in adjmats:
        w, embed = la.eigs(A=adjmat, k=min(d, adjmat.shape[0]-2), which='LM', tol=0)
        embed = np.absolute(embed)
        # fill out embedding with zeros if we couldn't get enough eigenvectors
        if embed.shape[1] < d:
            embed = np.concatenate((embed, np.zeros((embed.shape[0],(d-embed.shape[1])))), axis=1)
        embeddings.append(embed)
    for i in range(n):
        for j in range(i, n):
            if labeled:
                costs = label_dists(embeddings[i], embeddings[j], labels[i], labels[j])
            else:
                costs = ot.dist(embeddings[i], embeddings[j], metric='euclidean')
            M[i, j] = ot.emd2([], [], costs)
            M[j, i] = M[i, j]
    return M


def scemdkernel(adjmats, d=6, k=6, labeled=False, labels=None):
    n = len(adjmats)
    M = np.zeros((n, n))
    embeddings = []
    weights = []
    newlabels = []
    for adjmat in adjmats:
        # since we are clustering, don't need close tolerance
        w, embed = la.eigs(A=adjmat, k=min(d, adjmat.shape[0]-2), which='LM', tol=1 / (k + 1))
        embed = np.absolute(embed)
        # fill out embedding with zeros if we couldn't get enough eigenvectors
        if embed.shape[1] < d:
            embed = np.concatenate((embed, np.zeros((embed.shape[0], (d - embed.shape[1])))), axis=1)
        # get count for how many vertices appear in each slice
        counts = []
        newembeds = []
        inddict = dict()
        curind = 0
        for coord in embed:
            sliced = np.ceil(coord * k)
            slicestr = np.array2string(sliced)
            if slicestr in inddict:
                counts[inddict[slicestr]] += 1
            else:
                newembeds.append((sliced + 1/2)/k)
                counts.append(1)
                inddict[slicestr] = curind
                curind += 1
        # build weight and coord vectors for nonzero slices
        weights.append(np.array(counts)/np.sum(counts))
        newembeds = np.vstack(newembeds)
        embeddings.append(newembeds)
    for i in range(n):
        for j in range(i, n):
            if labeled:
                costs = label_dists_coarse(embeddings[i], embeddings[j], labels[i], labels[j])
            else:
                costs = ot.dist(embeddings[i], embeddings[j], metric='euclidean')
            M[i, j] = ot.emd2(weights[i], weights[j], costs)
            M[j, i] = M[i, j]
    return M
