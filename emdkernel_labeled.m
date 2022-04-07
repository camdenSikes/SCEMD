function [K, eigruntime,otherruntime] = emdkernel_labeled(Graphs, d)
% Computes the Earth Mover's Distance graph kernel for a set of labeled graphs
%
% Input:
%   Graphs: a 1 x N array of graphs
%   d: dimensionality of node embeddings
%
% Output:
%   K: the N x N kernel matrix
%   runtime: runtime in seconds
%
% Copyright (c) 2022, Camden Sikes

if nargin < 2
    d = 6;
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
    %for instances of small graphs (should be rare), pad out embeddings
    %with 0
    if(n<d)
        U = [U zeros(n,d-n)];
    end
    %append labels to the beginning
    U = [Graphs(i).nl.values U];
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
        n = size(Us{i},1);
        m = size(Us{j},1);
        [~,~,dist] = evalc('emd(Us{i},Us{j},ones(n,1)/n,ones(m,1)/m,@gdf_labeled)');
        K(i,j) = dist;
        K(j,i) = K(i,j);
    end
end
otherruntime = toc;
fprintf(1,'\n');
disp(['Kernel computation took ', num2str(otherruntime), ' sec']);
end

function dist = gdf_labeled(u,v)
% Computes the gdf between 2 vectors if they are labeled the same, otherwise
% returns sqrt(d) u and v are the vectors with the label added in the first
% position
    if(u(1) ~= v(1))
        dist = sqrt(length(u)-1);
    else
        dist = gdf(u(2:end),v(2:end));
    end
end