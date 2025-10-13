import heapq

def a_star_search(graph, heuristics, start, goal):
    open_list = [(heuristics[start], start, [start], 0)]
    closed_set = set()

    print("\nA* Search Traversal Order:", end=" ")

    while open_list:
        f, current, path, g = heapq.heappop(open_list)

        if current in closed_set:
            continue

        print(current, end=" ")
        closed_set.add(current)

        if current == goal:
            return path, g

        for neighbor, cost in graph[current]:
            if neighbor not in closed_set:
                g_new = g + cost
                f_new = g_new + heuristics[neighbor]
                heapq.heappush(open_list, (f_new, neighbor, path + [neighbor], g_new))

nodes = ['S', 'A', 'B', 'G']
graph = {}

print("Enter neighbors for each node ")
for node in nodes:
    edges = input(f"Enter edges from {node}: ")
    graph[node] = []
    if edges.strip():
        for edge in edges.split(","):
            n, c = edge.strip().split()
            graph[node].append((n, int(c)))

heuristics = {}
print("\nEnter heuristic values for each node:")
for node in nodes:
    heuristics[node] = int(input(f"h({node}) = "))

start_node = 'S'
goal_node = 'G'

path, cost = a_star_search(graph, heuristics, start_node, goal_node)
print(f"\n\nOptimal Path from {start_node} to {goal_node}: {path} with cost {cost}")
