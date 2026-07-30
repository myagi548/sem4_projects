import time

# ---------- Read Graph ----------
def read_graph_edges(filename):
    nodes = set()
    edges = []

    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        for _ in range(m):
            u, v, w = f.readline().split()
            w = int(w)
            nodes.add(u)
            nodes.add(v)
            edges.append((w, u, v))  
    
    return nodes, edges

# ---------- Merge Sort ----------
def merge_sort(edges):
    if len(edges) <= 1:
        return edges
    mid = len(edges) // 2
    left = merge_sort(edges[:mid])
    right = merge_sort(edges[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:  
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ---------- Kruskal's Algorithm ----------
def kruskal_mst(nodes, edges):
    mst = []
    total_weight = 0

    # Union-Find setup
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}

    # Sort edges by weight
    sorted_edges = merge_sort(edges)

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1

    for w, u, v in sorted_edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))
            total_weight += w

    return total_weight, mst

# ---------- Main ----------
if __name__ == "__main__":
    filename = "test1.txt"
    nodes, edges = read_graph_edges(filename)

    start = time.time()
    cost, mst = kruskal_mst(nodes, edges)
    end = time.time()

    print("Kruskal MST Cost:", cost)
    print("MST Edges:", mst)
    print("Runtime (seconds):", end - start)
