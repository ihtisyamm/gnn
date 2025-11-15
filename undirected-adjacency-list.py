'''
    We use an array of lists (or vector of lists) to represent the graph.
    The size of the array is equal to the number of vertices (here, 3).
    Each index in the array represents a vertex.
    Vertex 0 has two neighbours (1 and 2).
    Vertex 1 has two neighbours (0 and 2).
    Vertex 2 has two neighbours (0 and 1).
'''
def CreateGraph(V, edges):
    adj = [[] for _ in range(V)]

    for edge in edges:
        u = edge[0]
        v = edge[1]
        adj[u].append(v)

        # since undirected
        adj[v].append(u)
    return adj

V = 3

#List of edges (u,v)
edges = [[0, 1], [0, 2], [1, 2]]

#Build graph
adj = CreateGraph(V, edges)

print("Adjacency List Representation: ")
for i in range(V):
    #Print the vertex:
    print(f"{i}:", end=" ")
    for j in adj[i]:

        #print its adjacent
        print(j, end=" ")
    print()