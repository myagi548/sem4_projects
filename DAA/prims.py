import time
import heapq

# ---------- Read Graph ----------
def read_graph(filename):
    graph = {}
    nodes = set()

    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
    
        for _ in range(m):
            u, v, w = f.readline().split()
            w = int(w)
            nodes.add(u)
            nodes.add(v)

            graph.setdefault(u, []).append((w, v))
            graph.setdefault(v, []).append((w, u))

    return nodes, graph

# ---------- Prim's Algorithm ----------
def prim_mst(nodes, graph):
    start = next(iter(nodes))
    visited = {start}
    heap = []
    mst = []
    total_cost = 0

    for w, v in graph[start]:
        heapq.heappush(heap, (w, start, v))

    while heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(heap)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, w))
            total_cost += w

            for ew, nv in graph[v]:
                if nv not in visited:
                    heapq.heappush(heap, (ew, v, nv))

    return total_cost, mst

# ---------- Main ----------
if __name__ == "__main__":
    filename = "set50k.txt"
    nodes, graph = read_graph(filename)

    start = time.time()
    cost, mst = prim_mst(nodes, graph)
    end = time.time()

    print("Prim MST Cost:", cost)
    print("Runtime (seconds):", end - start)
