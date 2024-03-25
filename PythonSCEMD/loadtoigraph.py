import igraph as ig
import numpy as np
def loadtoigraph(directorypath,dataset):
    with open(directorypath+"/"+dataset+"_A.txt") as reader:
        edges = reader.readlines()
    with open(directorypath+"/"+dataset+"_graph_indicator.txt") as reader:
        graphind = [int(x) for x in reader.readlines()]
    with open(directorypath+"/"+dataset+"_graph_labels.txt") as reader:
        graphlab = [int(x) for x in reader.readlines()]
    with open(directorypath+"/"+dataset+"_node_labels.txt") as reader:
        nodelab = [int(x) for x in reader.readlines()]
    N = len(graphlab)
    n = len(nodelab)
    m = len(edges)
    nodelists = []
    graphs = []
    for i in range(1,N+1):
        #nodes = np.where(graphind == i)[]
        nodes = [j for j, x in enumerate(graphind) if x == i]
        nodelists.append(nodes)
        graph = ig.Graph(n = len(nodes))
        graphs.append(graph)
    for edge in edges:
        pair = edge.split(',')
        node1 = int(pair[0])-1
        node2 = int(pair[1])-1
        i = graphind[node1]-1
        node1id = nodelists[i].index(node1)
        node2id = nodelists[i].index(node2)
        graphs[i].add_edge(node1id, node2id)
    for i,lab in enumerate(nodelab):
        j = graphind[i]-1
        nodeid = nodelists[j].index(i)
        graphs[j].vs[nodeid]['label'] = lab
    return graphs,graphlab





if __name__ == "__main__":
    graphs,labels = loadtoigraph("/home/camden/Downloads/NCI1","NCI1")