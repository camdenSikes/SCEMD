% A simple test of the spectral clustering emd graph kernel, requires:
% libsvm, the runsvm file provided at https://members.cbio.mines-paristech.fr/~nshervashidze/code/
% and an implementation of the earth mover's distance
clearvars;
close all;
clc;

LIBSVMPATH = 'libsvm-3.25/matlab';

addpath('svm');
addpath(LIBSVMPATH);

datapath = 'datasets/MUTAG.mat';
load(datapath);
dataset = MUTAG;
labels = lmutag;

rng(27);

tic;
N = size(dataset,2);
numiter = 1;

Ws = cell(1,N);

[K,~,~] = specclus_emdkernel_unlabeled(dataset, 6 , 32  );

try chol(K);
    disp('Is positive semidefinite')
    Ks{1} = K;
    result = runsvm(Ks, labels);
catch ME
    disp("Isn't positive semidefinite")
    maxval = max(K,[],'all');
    K = exp(-K/maxval);
    Ks{1} = K;
    
    result = runsvm(Ks, labels);
   
end


