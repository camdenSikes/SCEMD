function [K, eigruntime,otherruntime] = weighted_emdkernel_unlabeled(Graphs, Weights, d, option)
% Computes the modified earth movers distance graph kernel for a set of 
% unlabeled graphs, where each node has a corresponding weight, perhaps
% generated via coarsening from larger graphs
%
% Input:
%   Graphs: a 1 x N array of graphs
%   Weights: a 1 x N cell array of vertex weight vectors
%   matrices and vertex weights
%   d: dimensionality of node embeddings
%   option: 'accurate' - full earth mover's distance computation
%           'estimate' - emd estimation
%           'even' - even flow between every pair of vertices
%
% Output:
%   K: the N x N kernel matrix
%   runtime: runtime in seconds
% 
% Copyright (c) 2022, Camden Sikes

if nargin < 3
    d = 6;
end

if nargin < 4
    option = "accurate";
end

N = size(Graphs,2);
tic;

% Compute embeddings
disp('Computing embeddings...');
Us = cell(1,N);
for i=1:N
    n = size(Graphs(i).am, 1);
    [U, ~] = eigs(Graphs(i).am, min(n, d));
    U = abs(U);
    Us{i} = U;
end

eigruntime = toc;
fprintf(1,'\n');
disp(['eigenvector computation took ', num2str(eigruntime), ' sec']);
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
        wi = sum(Weights{i});
        wj = sum(Weights{j});
        
        if i == j
            dist = 0;
        elseif option == "accurate"
            [~,~,dist] = evalc("emd(Us{i},Us{j},Weights{i}/wi,Weights{j}/wj,@gdf)");
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