function [A,vertices,weights] = spectralcoarsen_labeled(Graph,d,k,buildMat,maxlabelval)
% Computes a coarsened version of a graph inspired by the pyramid match
% graph kernel. Each slice represents a potential vertex, weighted by how
% many vertices appear in that slice. Vertices are connected if vertices in
% those slices are connected
% Input:
%   Graph: the graph to coarsen
%   d: the number of dimensions/number of eigenvectors to embed
%   k: the level of coarsening. Space is split into k slices per
%   dimension
%   buildMat: True if A should be created, false otherwise
%   maxlabelval: The highest possible value for a label to be
%
% Output:
%   A: adjacency matrix of coarsened graph, edges weighted by number of
%   representatives
%   vertices: vertices with coordinates representing the region of the 
%   eigenspace the vertex represents and its multilabel
%   weights: number of vertices represented in coarsened vertex
%
% Copyright (c) 2022, Camden Sikes
G = Graph.am;
N = size(G,1);
d = min(d,N);
if(nargin < 5)
    maxlabelval = max(Graph.nl.values);
end
[eigvecs,~] = eigs(G,min(d,N),"largestreal");
eigvecs = abs(eigvecs);
%for instances of small graphs (should be rare), pad out embeddings
%with 0
if(N<d)
    eigvecs = [eigvecs zeros(N,d-N)];
end

%Get count of how many vertices appear in each slice
counts = containers.Map;
labelMap = containers.Map;
slices = zeros(N,d);
for ind = 1:N
    coord = eigvecs(ind,:);
    slice = ceil(coord*(k));
    slices(ind,:) = slice;
    if(isKey(counts,num2str(slice)))
        counts(num2str(slice)) = counts(num2str(slice)) + 1;
        label = labelMap(num2str(slice));
        label(Graph.nl.values(ind)) = label(Graph.nl.values(ind)) + 1;
        labelMap(num2str(slice)) = label;
    else
        counts(num2str(slice)) = 1;
        label = zeros(maxlabelval,1);
        label(Graph.nl.values(ind)) = 1;
        labelMap(num2str(slice)) = label;
    end
end

%build weight and coordinate vectors for the non-zero slices
n = length(counts);
slice2ind = containers.Map;
vertices.coords = zeros(1,d);
vertices.label = zeros(1,maxlabelval);
weights = zeros(n,1);
curind = 1;
for slice = keys(counts)
    slice = string(slice);
    slice2ind(slice) = curind;
    center = str2num(slice)/k - (1/(2*k));
    vertices(curind,1).coords = center;
    vertices(curind,1).label = labelMap(slice);
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