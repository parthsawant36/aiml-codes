import heapq
def greedy_best_first_search(graph, heuristics, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristics[start], start))  # (heuristic, node)

    visited = set()
    parent = {start: None}  # To reconstruct path

    print("\n--- Search Process ---")
    while open_list:
        h, current = heapq.heappop(open_list)
        print(f"Expanding Node: {current} (h={h})")

        if current == goal:
            print("\nGoal Reached!")
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]  # Reverse path

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                heapq.heappush(open_list, (heuristics[neighbor], neighbor))
                parent[neighbor] = current

    return None  # Goal not reachable
graph = {}
print("Enter Graph Details")
n = int(input("Enter number of nodes: "))

nodes = []
for i in range(n):
    node = input(f"Enter node name ({i + 1}): ").strip()
    nodes.append(node)

print("\nEnter neighbors for each node (comma separated).")
print("Example: For A -> B,C   (leave blank if no neighbor)")
for node in nodes:
    neighbors = input(f"Neighbors of {node}: ").strip()
    graph[node] = [x.strip() for x in neighbors.split(",")] if neighbors else []

heuristics = {}
print("\nEnter heuristic (h) values for each node:")
for node in nodes:
    h = int(input(f"Heuristic value for {node}: "))
    heuristics[node] = h

start = input("\nEnter Start Node: ").strip()
goal = input("Enter Goal Node: ").strip()


path = greedy_best_first_search(graph, heuristics, start, goal)

print("\n--- Result ---")
if path:
    print("Path Found: ", " -> ".join(path))
else:
    print("No path exists from", start, "to", goal)


