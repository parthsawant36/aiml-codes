from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    traversal_order = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            # Use graph.get(node, []) to avoid KeyError
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return traversal_order


# ----- User Input -----
graph = {}
n = int(input("Enter number of nodes: "))

for i in range(n):
    node = input(f"Enter node name {i+1}: ")
    neighbors = input(f"Enter neighbors of {node} separated by space: ").split()
    graph[node] = neighbors

start = input("Enter starting node for BFS: ")

print("\nBreadth First Search Traversal:")
print(" -> ".join(bfs(graph, start)))
