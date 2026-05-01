import sys
from collections import deque


def read_graph(graph_file):
    with open(graph_file, "r") as f:
        lines = f.readlines()

    num_nodes, num_edges = map(int, lines[0].split())

    edges = []
    for line in lines[1:1 + num_edges]:
        u, v = map(int, line.split())
        edges.append((u, v))

    alice_start, bob_start = map(int, lines[1 + num_edges].split())

    return num_nodes, num_edges, edges, alice_start, bob_start


def build_graph(num_nodes, edges, directed):
    graph = [[] for _ in range(num_nodes)]

    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)

    for neighbors in graph:
        neighbors.sort()

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


def find_meeting_node(num_nodes, alice_dist, bob_dist):
    best_node = -1
    best_dist = float("inf")
    best_parity = -1

    for node in range(num_nodes):
        for parity in [0, 1]:
            da = alice_dist[node][parity]
            db = bob_dist[node][parity]

            if da != -1 and db != -1 and da == db:
                if da < best_dist:
                    best_node = node
                    best_dist = da
                    best_parity = parity
                elif da == best_dist and node > best_node:
                    best_node = node
                    best_parity = parity

    return best_node, best_dist, best_parity


def reconstruct_path(parent, node, parity):
    path = []
    current_node = node
    current_parity = parity

    while current_node != -1:
        path.append(current_node)
        current_node, current_parity = parent[current_node][current_parity]

    path.reverse()
    return path


def print_meeting(alice_path, bob_path, meeting_node):
    for i in range(len(alice_path)):
        print(f"{i}: Alice at {alice_path[i]}, Bob at {bob_path[i]}")

    print(f"Meeting at node {meeting_node} at time step {len(alice_path) - 1}.")


def get_meeting_solution(graph, num_nodes, alice_start, bob_start):
    alice_dist, alice_parent = bfs_with_parity(graph, alice_start)
    bob_dist, bob_parent = bfs_with_parity(graph, bob_start)

    meeting_node, meeting_dist, parity = find_meeting_node(
        num_nodes, alice_dist, bob_dist
    )

    if meeting_node == -1:
        return None

    alice_path = reconstruct_path(alice_parent, meeting_node, parity)
    bob_path = reconstruct_path(bob_parent, meeting_node, parity)

    return meeting_node, alice_path, bob_path


def solve_current_graph(graph, num_nodes, alice_start, bob_start):
    solution = get_meeting_solution(graph, num_nodes, alice_start, bob_start)

    if solution is None:
        return False

    meeting_node, alice_path, bob_path = solution
    print_meeting(alice_path, bob_path, meeting_node)
    return True


def bfs_path(graph, start, target):
    n = len(graph)
    visited = [False] * n
    parent = [-1] * n

    q = deque()
    q.append(start)
    visited[start] = True

    while q:
        node = q.popleft()

        if node == target:
            break

        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                q.append(neighbor)

    if not visited[target]:
        return []

    path = []
    current = target

    while current != -1:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def add_edge(graph, u, v, directed):
    if v not in graph[u]:
        graph[u].append(v)
        graph[u].sort()

    if not directed:
        if u not in graph[v]:
            graph[v].append(u)
            graph[v].sort()


def fix_undirected_graph(graph, alice_start, bob_start):
    path = bfs_path(graph, alice_start, bob_start)

    if not path:
        return []

    if len(path) == 2:
        a = path[0]
        b = path[1]

        for neighbor in graph[b]:
            if neighbor != a:
                return [(a, neighbor)]

        return []

    middle_index = len(path) // 2
    meeting_node = path[middle_index]
    previous_node = path[middle_index - 2]

    return [(previous_node, meeting_node)]


def simple_distances(graph, start):
    n = len(graph)
    dist = [-1] * n

    q = deque()
    q.append(start)
    dist[start] = 0

    while q:
        node = q.popleft()

        for neighbor in graph[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                q.append(neighbor)

    return dist


def find_directed_base_node(graph, num_nodes, alice_start, bob_start):
    alice_dist = simple_distances(graph, alice_start)
    bob_dist = simple_distances(graph, bob_start)

    best_node = -1
    best_sum = float("inf")

    for node in range(num_nodes):
        if alice_dist[node] != -1 and bob_dist[node] != -1:
            total = alice_dist[node] + bob_dist[node]

            if total < best_sum:
                best_sum = total
                best_node = node

    return best_node


def fix_directed_graph_one_edge(graph, num_nodes, alice_start, bob_start):
    u = find_directed_base_node(graph, num_nodes, alice_start, bob_start)

    if u == -1:
        return [], None

    best_edge = None
    best_solution = None
    best_time = float("inf")

    # 2-cycle candidates: add u -> w where w -> u already exists
    for w in range(num_nodes):
        if u in graph[w] and w not in graph[u]:
            new_graph = [neighbors[:] for neighbors in graph]
            add_edge(new_graph, u, w, True)

            solution = get_meeting_solution(
                new_graph, num_nodes, alice_start, bob_start
            )

            if solution is not None:
                meeting_node, alice_path, bob_path = solution
                time_step = len(alice_path) - 1

                if time_step < best_time:
                    best_time = time_step
                    best_edge = (u, w)
                    best_solution = solution

    if best_edge is None:
        return [], None

    return [best_edge], best_solution


def print_added_edges(edges_to_add):
    if len(edges_to_add) == 1:
        print("Adding 1 edge.")
    else:
        print(f"Adding {len(edges_to_add)} edges.")

    for u, v in edges_to_add:
        print(f"Adding {u} {v}.")


def main():
    args = sys.argv
    directed = False

    if len(args) < 2:
        return

    if args[1] == "-d":
        directed = True
        graph_file = args[2]
    else:
        graph_file = args[1]

    num_nodes, num_edges, edges, alice_start, bob_start = read_graph(graph_file)
    graph = build_graph(num_nodes, edges, directed)

    if solve_current_graph(graph, num_nodes, alice_start, bob_start):
        return

    print("No meeting is possible.")

    if directed:
        edges_to_add, solution = fix_directed_graph_one_edge(
            graph, num_nodes, alice_start, bob_start
        )

        if not edges_to_add:
            print("Could not establish a rendezvous by adding edges.")
            return

        print_added_edges(edges_to_add)

        meeting_node, alice_path, bob_path = solution
        print_meeting(alice_path, bob_path, meeting_node)
        return

    edges_to_add = fix_undirected_graph(graph, alice_start, bob_start)

    if not edges_to_add:
        print("Could not establish a rendezvous by adding edges.")
        return

    print_added_edges(edges_to_add)

    for u, v in edges_to_add:
        add_edge(graph, u, v, directed)

    if not solve_current_graph(graph, num_nodes, alice_start, bob_start):
        print("Could not establish a rendezvous by adding edges.")


if __name__ == "__main__":
    main()