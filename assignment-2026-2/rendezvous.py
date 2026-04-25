import sys


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

    print("Directed:", directed)
    print("Nodes:", num_nodes)
    print("Edges:", num_edges)
    print("Alice:", alice_start)
    print("Bob:", bob_start)
    print("Edge list:", edges)


if __name__ == "__main__":
    main()