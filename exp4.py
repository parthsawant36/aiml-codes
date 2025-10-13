def dfs(graph, start, goal):
    visited = set()
    stack = [(start, [start])]
    print("DFS Traversal Order:", end=" ")
    while stack:
        node, path = stack.pop()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            if node == goal:
                return path
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

nodes = ['S', 'A', 'B', 'C', 'D', 'G']
graph = {}

print("Enter neighbors for each node (comma-separated). Example: A,B")
for node in nodes:
    edges = input(f"Enter edges from {node}: ")
    graph[node] = []
    if edges.strip():
        graph[node] = [n.strip() for n in edges.split(",")]

start_node = 'S'
goal_node = 'G'

path = dfs(graph, start_node, goal_node)
print("\nPath from {} to {}: {}".format(start_node, goal_node, path if path else "No path found"))
