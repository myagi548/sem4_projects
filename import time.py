import time

# ---------- Read graph ----------
def read_graph(filename):
    edges = []
    nodes = set()
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        for _ in range(m):
            u, v, w = f.readline().split()
            w = int(w)
            edges.append((w, u, v))  # store as (weight, u, v)
            nodes.add(u)
            nodes.add(v)
    return nodes, edges

# ---------- Union-Find ----------
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    parent[find(parent, x)] = find(parent, y)

# ---------- Kruskal's Algorithm ----------
def kruskal_mst(nodes, edges):
    parent = {node: node for node in nodes}
    mst = []
    total_weight = 0

    edges.sort()  # sort edges by weight

    for w, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, u, v)
            mst.append((u, v, w))
            total_weight += w

    return total_weight, mst

# ---------- Main ----------
if __name__ == "__main__":
    filename = "set5.txt"  # change dataset here
    nodes, edges = read_graph(filename)

    start_time = time.time()
    total_weight, mst = kruskal_mst(nodes, edges)
    end_time = time.time()

    runtime = end_time - start_time

    print("Kruskal MST Total Weight:", total_weight)
    print("Edges in MST:", mst)
    print("Runtime (seconds):", runtime)
