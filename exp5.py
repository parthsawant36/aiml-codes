import heapq

def greedy_best_first_search(graph, heuristics, start, goal):
    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, (heuristics[start], start, [start]))  # (h(n), node, path)

    print("\nGreedy Best-First Search Traversal Order:", end=" ")

    while priority_queue:
        h, current_node, path = heapq.heappop(priority_queue)

        
        if current_node not in visited:
            print(current_node, end=" ")
            visited.add(current_node)

           
            if current_node == goal:
                return path

            
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristics[neighbor], neighbor, path + [neighbor]))

    return None

nodes = ['S', 'A', 'B', 'G']
graph = {}

print("Enter neighbors for each node (comma-separated). Example: A,B")
for node in nodes:
    edges = input(f"Enter edges from {node}: ")
    graph[node] = []
    if edges.strip():
        graph[node] = [n.strip() for n in edges.split(",")]

heuristics = {}
print("\nEnter heuristic values for each node:")
for node in nodes:
    heuristics[node] = int(input(f"h({node}) = "))

start_node = 'S'
goal_node = 'G'

path = greedy_best_first_search(graph, heuristics, start_node, goal_node)
print("\nPath from {} to {}: {}".format(start_node, goal_node, path if path else "No path found"))
