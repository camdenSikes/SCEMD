function [A,coords,weights] = spectralcoarsen(G,d,k,buildMat)
% Computes a coarsened version of a graph inspired by the pyramid match
% graph kernel. Each slice represents a potential vertex, weighted by how
% many vertices appear in that slice. Vertices are connected if vertices in
% those slices are connected
% Input:
%   G: the graph to coarsen
%   d: the number of dimensions/number of eigenvectors to embed
%   k: the level of coarsening. Space is split into k slices per
%   dimension
%   buildMat: True if A should be created, false otherwise
%
%Output:
%   A: adjacency matrix of coarsened graph, edges weighted by number of
%   representatives
%   coords: coordinates representing the region of the eigenspace the
%   vertex represents
%   weights: number of vertices represented in coarsened vertex
%
% Copyright (c) 2022, Camden Sikes
N = size(G,1);
d = min(d,N);
[eigvecs,~] = eigs(G,d,"largestreal");
eigvecs = abs(eigvecs);

%Get count of how many vertices appear in each slice
counts = containers.Map;
slices = zeros(N,d);
for ind = 1:N
    coord = eigvecs(ind,:);
    slice = ceil(coord*(k));
    slices(ind,:) = slice;
    if(isKey(counts,num2str(slice)))
        counts(num2str(slice)) = counts(num2str(slice)) + 1;
    else
        counts(num2str(slice)) = 1;
    end
end

%build weight and coordinate vectors for the non-zero slices
n = length(counts);
slice2ind = containers.Map;
coords = zeros(n,d);
weights = zeros(n,1);
curind = 1;
for slice = keys(counts)
    slice = string(slice);
    slice2ind(slice) = curind;
    center = str2num(slice)/k - (1/(2*k));
    coords(curind,:) = center;
    weights(curind) = counts(slice);
    curind = curind + 1;
end
if(buildMat)
    %build weighted adjacency matrix
    A = zeros(n,n);
    [row,col] = find(G);
    for i = 1:size(row)
        if(row(i) > col(i))
            source = slice2ind(num2str(slices(row(i),:)));
            sink = slice2ind(num2str(slices(col(i),:)));
            A(source,sink) = A(source,sink) + 1;
        end
    end
else
    A = 0;
end


end