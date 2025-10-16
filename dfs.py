def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()          # To keep track of visited nodes

    if start not in visited:
        visited.add(start)
        print(start, end=" ")    # Print the node when visited
        # Explore each neighbor
        for neighbor in graph.get(start, []):   # graph.get() avoids KeyError
            dfs(graph, neighbor, visited)

graph = {}
n = int(input("Enter number of nodes: "))

# Taking graph input as adjacency list
for i in range(n):
    node = input(f"Enter node name {i+1}: ")
    neighbors = input(f"Enter neighbors of {node} separated by space: ").split()
    graph[node] = neighbors

start = input("Enter starting node for DFS: ")

print("\nDepth First Search Traversal:")
dfs(graph, start)
print()

