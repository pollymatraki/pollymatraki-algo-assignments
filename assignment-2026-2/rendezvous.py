import sys


def main():
    args = sys.argv

    directed = False

    if len(args) < 2:
        print("Usage: python rendezvous.py [-d] <graph_file>")
        return

    if args[1] == "-d":
        directed = True
        graph_file = args[2]
    else:
        graph_file = args[1]

    print("Directed:", directed)
    print("Graph file:", graph_file)

    # 🔹 ΝΕΟ ΚΟΜΜΑ
    try:
        with open(graph_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("File not found")
        return

    print("Total lines:", len(lines))


if __name__ == "__main__":
    main()