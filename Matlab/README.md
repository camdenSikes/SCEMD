# SCEMD
Code for the Spectral Clustering Earth Mover's Distance Graph Kernel
# Dependencies
To generate the kernels, you need to download an implementation of the earth mover's distance, for example the one by Ulas Yilmaz available at https://www.mathworks.com/matlabcentral/fileexchange/22962-the-earth-mover-s-distance

To use the kernels to train SVMs, I recommend installing LIBSVM and using the svm folder available at https://members.cbio.mines-paristech.fr/~nshervashidze/code/ This site also supplies some datasets.
# Files
*spectralcoarsen\[_labeled\].m* contains the code to embed the vertices of the graph and split them into weighted buckets. If the BuildMat setting is selected, it also assigns edges between the embedded nodes (this takes significant extra time).

*specclus_emdkernel_\[un\]labeled.m* build the graph kernel matrices for datasets of labeled or unlabeled graphs

*emdkernel_\[un\]labeled.m* are implementations of the emd graph kernel proposed in the paper by Nikolentzos et. al.

*emdkernel_weighted.m* is a modified emd graph kernel which allows weights to be associated with vertices (can be used if you coarsen the graphs before applying the kernel).

*example.m* is a basic example of the use of the kernel.
