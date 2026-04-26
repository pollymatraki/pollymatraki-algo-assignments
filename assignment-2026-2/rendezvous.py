import sys
from collections import deque


def read_graph(graph_file):
    with open(graph_file, "r") as f:
        lines = f.readlines()

    first_line = lines[0].split()
    num_nodes = int(first_line[0])
    num_edges = int(first_line[1])

    edges = []

    for line in lines[1:1 + num_edges]:
        parts = line.split()
        x = int(parts[0])
        y = int(parts[1])
        edges.append((x, y))

    last_line = lines[1 + num_edges].split()
    alice_start = int(last_line[0])
    bob_start = int(last_line[1])

    return num_nodes, num_edges, edges, alice_start, bob_start

def build_graph(num_nodes, edges, directed):
    graph = [[] for _ in range(num_nodes)]

    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)

    return graph

def bfs_with_parity(graph, start):
    n = len(graph)

    dist = [[-1, -1] for _ in range(n)]
    parent = [[None, None] for _ in range(n)]

    q = deque()
    q.append((start, 0))

    dist[start][0] = 0
    parent[start][0] = (-1, -1)

    while q:
        node, parity = q.popleft()

        for neighbor in graph[node]:
            new_parity = 1 - parity

            if dist[neighbor][new_parity] == -1:
                dist[neighbor][new_parity] = dist[node][parity] + 1
                parent[neighbor][new_parity] = (node, parity)
                q.append((neighbor, new_parity))

    return dist, parent

def main():
    args = sys.argv
    directed = False

    if len(args) < 2:
        print("Usage: py rendezvous.py [-d] <graph_file>")
        return

    if args[1] == "-d":
        directed = True
        graph_file = args[2]
    else:
        graph_file = args[1]

    try:
        num_nodes, num_edges, edges, alice_start, bob_start = read_graph(graph_file)
    except FileNotFoundError:
        print("File not found")
        return


    graph = build_graph(num_nodes, edges, directed)
    print("Adjacency list:")
    for i in range(num_nodes):
        print(i, "->", graph[i])
        
    print("Directed:", directed)
    print("Nodes:", num_nodes)
    print("Edges:", num_edges)
    print("Alice:", alice_start)
    print("Bob:", bob_start)
    print("Edge list:", edges)

    alice_dist, alice_parent = bfs_with_parity(graph, alice_start)
    bob_dist, bob_parent = bfs_with_parity(graph, bob_start)

    print("Alice distances:", alice_dist)
    print("Bob distances:", bob_dist)


if __name__ == "__main__":
    main()