function dist = emdeven(g1, g2)
% Computes the cost of the flow if there is an even amount of flow
% Between every pair of embedded nodes
    n = size(g1,1);
    m = size(g2,1);
    dist = 0;
    for i=1:n
        for j=1:m
            dist = dist + 1/(n*m)*norm(g1(i)-g2(j));
        end
    end
end