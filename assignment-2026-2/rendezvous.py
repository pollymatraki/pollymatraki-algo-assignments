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


def add_edge(graph, u, v, directed):
    if v not in graph[u]:
        graph[u].append(v)
        graph[u].sort()

    if not directed:
        if u not in graph[v]:
            graph[v].append(u)
            graph[v].sort()


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


def fix_undirected_graph(graph, alice_start, bob_start):
    path = bfs_path(graph, alice_start, bob_start)

    if not path:
        return []

    if len(path) == 2:
        a = path[0]
        b = path[1]

        for neighbor in graph[a]:
            if neighbor != b:
                return [(b, neighbor)]

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


def directed_base_nodes(graph, num_nodes, alice_start, bob_start):
    alice_dist = simple_distances(graph, alice_start)
    bob_dist = simple_distances(graph, bob_start)

    candidates = []

    for node in range(num_nodes):
        if alice_dist[node] != -1 and bob_dist[node] != -1:
            candidates.append((alice_dist[node] + bob_dist[node], node))

    candidates.sort()
    return [node for _, node in candidates]

def get_meeting_solution_by_states(graph, alice_start, bob_start):
    start_state = (alice_start, bob_start)

    q = deque()
    q.append(start_state)

    visited = set()
    visited.add(start_state)

    parent = {start_state: None}

    while q:
        alice, bob = q.popleft()

        if alice == bob:
            states = []
            current = (alice, bob)

            while current is not None:
                states.append(current)
                current = parent[current]

            states.reverse()

            alice_path = [state[0] for state in states]
            bob_path = [state[1] for state in states]

            return alice, alice_path, bob_path

        for next_alice in graph[alice]:
            for next_bob in graph[bob]:
                next_state = (next_alice, next_bob)

                if next_state not in visited:
                    visited.add(next_state)
                    parent[next_state] = (alice, bob)
                    q.append(next_state)

    return None

def directed_two_cycle_edges(graph, num_nodes, u):
    edges = []

    for w in range(num_nodes):
        if u in graph[w] and w not in graph[u]:
            edges.append((u, w))

    return edges


def directed_three_cycle_edges(graph, num_nodes, u):
    edges = []

    for w in range(num_nodes):
        if w == u:
            continue

        if w in graph[u]:
            continue

        for v in graph[w]:
            if u in graph[v]:
                edges.append((u, w))
                break

    return edges


def test_directed_edges(graph, num_nodes, alice_start, bob_start, edges_to_add):
    new_graph = [neighbors[:] for neighbors in graph]

    for u, v in edges_to_add:
        add_edge(new_graph, u, v, True)

    solution = get_meeting_solution_by_states(
        new_graph, alice_start, bob_start
    )

    return solution


def fix_directed_graph_one_edge(graph, num_nodes, alice_start, bob_start):
    base_nodes = directed_base_nodes(graph, num_nodes, alice_start, bob_start)

    for u in base_nodes:
        two_edges = directed_two_cycle_edges(graph, num_nodes, u)
        three_edges = directed_three_cycle_edges(graph, num_nodes, u)

        # Πρώτα δοκιμάζουμε έναν 2-cycle
        for edge in two_edges:
            solution = test_directed_edges(
                graph, num_nodes, alice_start, bob_start, [edge]
            )

            if solution is not None:
                return [edge], solution

        # Μετά δοκιμάζουμε έναν 3-cycle
        for edge in three_edges:
            solution = test_directed_edges(
                graph, num_nodes, alice_start, bob_start, [edge]
            )

            if solution is not None:
                return [edge], solution

        # Τέλος δοκιμάζουμε συνδυασμό 2-cycle + 3-cycle
        for edge1 in two_edges:
            for edge2 in three_edges:
                if edge1 == edge2:
                    continue

                solution = test_directed_edges(
                    graph, num_nodes, alice_start, bob_start, [edge1, edge2]
                )

                if solution is not None:
                    return [edge1, edge2], solution

    return [], None

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