import time
from itertools import combinations

# ---------- Read graph ----------
def read_graph(filename):
    edges = []
    nodes = set()
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        for _ in range(m):
            u, v, w = f.readline().split()
            w = int(w)
            edges.append((u, v, w))
            nodes.add(u)
            nodes.add(v)
    return nodes, edges

# ---------- Union-Find ----------
def find(parent, x):
    while parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(parent, x, y):
    parent[find(parent, x)] = find(parent, y)

# ---------- Check if subset is a spanning tree ----------
def is_spanning_tree(edge_subset, nodes):
    parent = {node: node for node in nodes}
    for u, v, _ in edge_subset:
        pu = find(parent, u)
        pv = find(parent, v)
        if pu == pv:  # cycle detected
            return False
        union(parent, pu, pv)
    # check if all nodes are connected
    root = find(parent, next(iter(nodes)))
    for node in nodes:
        if find(parent, node) != root:
            return False
    return True

# ---------- Brute Force MST ----------
def brute_force_mst(nodes, edges):
    n = len(nodes) 
    min_weight = float('inf')
    best_tree = None

    for subset in combinations(edges, n - 1):
        if is_spanning_tree(subset, nodes):
            weight = sum(w for _, _, w in subset)
            if weight < min_weight:
                min_weight = weight
                best_tree = subset

    return min_weight, best_tree

# ---------- Main ----------
if __name__ == "__main__":
    filename = "test5.txt"  
    nodes, edges = read_graph(filename)

    start_time = time.time()
    weight, mst = brute_force_mst(nodes, edges)
    end_time = time.time()

    runtime = end_time - start_time

    print("Brute Force MST Total Weight:", weight)
    print("Edges in MST:", mst)
    print("Runtime (seconds):", runtime)
