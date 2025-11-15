'''
    We use an array of lists (or vector of lists) to represent the graph.
    The size of the array is equal to the number of vertices (here, 3).
    Each index in the array represents a vertex.
    Vertex 0 has no neighbours
    Vertex 1 has two neighbours (0 and 2).
    Vertex 2 has 1 neighbours (0).
'''
def CreateGraph(V, edges):
    adj = [[] for _ in range(V)]

    # only loop the row
    # so row first is [0,1] then u=0, v=1
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adj[u].append(v)    # since directed, no vice-versa

    return adj

V = 3

#List of edges (u,v)
edges = [[0, 1], [0, 2], [1, 2]]

#Build graph
adj = CreateGraph(V, edges)

print("Adjacency directed list: ")
for i in range(V):

    print(f"{i}: ", end='')
    for j in adj[i]:
        print(j, end=" ")
    print()
