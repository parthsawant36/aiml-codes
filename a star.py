import heapq

def a_star_search(graph, heuristics, start, goal):
    # Priority Queue (min-heap) -> (f, g, node, path)
    open_list = []
    heapq.heappush(open_list, (heuristics[start], 0, start, [start]))  # f, g, node, path
    visited = set()

    print("\n--- Search Process ---")
    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        print(f"Expanding Node: {current} | g={g}, h={heuristics[current]}, f={f}")

        if current == goal:
            print("\nGoal Reached!")
            return path, g  # Return path and total cost

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                g_new = g + cost
                f_new = g_new + heuristics[neighbor]
                heapq.heappush(open_list, (f_new, g_new, neighbor, path + [neighbor]))

    return None, None

graph = {}
print("Enter Graph Details")
n = int(input("Enter number of nodes: "))

nodes = []
for i in range(n):
    node = input(f"Enter node name ({i+1}): ").strip()
    nodes.append(node)

print("\nEnter neighbors and cost for each node.")
print("Example: For A -> B(2),C(3)   (leave blank if no neighbor)")
for node in nodes:
    neighbors = input(f"Neighbors of {node}: ").strip()
    edges = []
    if neighbors:
        for item in neighbors.split(","):
            neigh, cost = item.strip().split("(")
            cost = int(cost.strip(")"))
            edges.append((neigh.strip(), cost))
    graph[node] = edges

heuristics = {}
print("\nEnter heuristic (h) values for each node:")
for node in nodes:
    h = int(input(f"Heuristic value for {node}: "))
    heuristics[node] = h

start = input("\nEnter Start Node: ").strip()
goal = input("Enter Goal Node: ").strip()

path, cost = a_star_search(graph, heuristics, start, goal)

print("\n--- Result ---")
if path:
    print("Optimal Path: ", " -> ".join(path))
    print("Total Cost: ", cost)
else:
    print("No path exists from", start, "to", goal)

