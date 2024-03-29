function [K, coarsenruntime,otherruntime] = specclus_emdkernel_labeled(Graphs, d, k)
% Computes the Spectral Clustering Earth Mover's Distance graph kernel of   
% a set of labeled graphs, by first clustering the graph embeddings and  
% setting the weights based on vertex density.
%
% Input:
%   Graphs: a 1 x N array of graphs
%   d: dimensionality of node embeddings
%   k: split each dimension of the eigenspace into k slices
%
% Output:
%   K: the N x N kernel matrix
%   runtime: runtime in seconds
%
% Copyright (c) 2022, Camden Sikes

if nargin < 2
    d = 6;
end
if nargin < 3
    k = 4;
end


N = size(Graphs,2);
tic;

maxlabelval = 0;
for i=1:N
    maxlabelval = max(maxlabelval,max(Graphs(i).nl.values));
end

    
% Coarsen, compute embeddings and weights
disp('Coarsening...');
Us = cell(1,N);
Ws = cell(1,N);

fprintf(1,'%s','Progress: 0%');
for i=1:N
    last_percent = floor(100*(i-1)/N);
    cur_percent = floor(100*i/N);
    
    if last_percent ~= cur_percent
        last_str = sprintf('%d%s',last_percent,'%');
        fprintf(1,repmat('\b', 1, length(last_str)));
        fprintf(1,'%d%s',cur_percent,'%');
    end
    
    [~,vertices,weights] = spectralcoarsen_labeled(Graphs(i),d,k,false,maxlabelval);
    
    Us{i} = vertices;
    Ws{i} = weights;
end

coarsenruntime = toc;
fprintf(1,'\n');
disp(['coarsening computation took ', num2str(coarsenruntime), ' sec']);
tic;

% Earth Mover's Distance computation
fprintf(1,'\n');
disp('Computing kernel matrix...');
K = zeros(N,N);
fprintf(1,'%s','Progress: 0%');
for i=1:N
    last_percent = floor(100*(i-1)/N);
    cur_percent = floor(100*i/N);
    
    if last_percent ~= cur_percent
        last_str = sprintf('%d%s',last_percent,'%');
        fprintf(1,repmat('\b', 1, length(last_str)));
        fprintf(1,'%d%s',cur_percent,'%');
    end
    
    for j=i:N
        wi = sum(Ws{i});
        wj = sum(Ws{j});
        
        if option == "accurate"
            [~,~,dist] = evalc('emd(Us{i},Us{j},Ws{i}/wi,Ws{j}/wj,@gdf_multilabeled)');
        elseif option == "even"
            dist = emdeven(Us{i},Us{j});
        end
        K(i,j) = dist;
        K(j,i) = K(i,j);
    end
end
otherruntime = toc;
fprintf(1,'\n');
disp(['Kernel computation took ', num2str(otherruntime), ' sec']);
end

function dist = gdf_multilabeled(u,v)
% Computes a weighted average of the distance between 2 vectors and sqrt(d) 
% where the weights are determined by the similarity of the multilabels
% u.vec is the vector and u.lab is the multilabel
% position
    alpha = 1;
    ulabeldist = u.label/sum(u.label);
    vlabeldist = v.label/sum(v.label);
    dist = alpha*gdf(u.coords,v.coords) + gdf(ulabeldist,vlabeldist);
end