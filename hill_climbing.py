import copy
import random
import time

class Greedy_Search:
    def __init__(self, num_vertices, capacity, distance):
        self.num_vertices = num_vertices
        self.capacity = capacity
        self.distance = distance
        self.pickup = list(range(1, num_vertices // 2 + 1))
        self.dropoff = list(range(num_vertices // 2 + 1, num_vertices + 1))
        
    def children(self, parent_state):
        current_location = parent_state[-1] if parent_state else 0
        current_capacity = self.current_capacity(parent_state)
        
        possible_locations = []
        
        for location in self.pickup + self.dropoff:
            if location in parent_state:
                continue
                
            if location in self.pickup:
                if current_capacity + 1 <= self.capacity:
                    possible_locations.append(location)
            else:  
                corresponding_pickup = location - self.num_vertices // 2
                if corresponding_pickup in parent_state:
                    possible_locations.append(location)
        
        return possible_locations
    
    def current_capacity(self, state):
        capacity = 0
        for location in state:
            if location in self.pickup:
                capacity += 1
            else:
                capacity -= 1
        return capacity
    
    def heuristic_greedy_function(self, current_location, next_location):
        return self.distance[current_location][next_location]
    
    def choose_node(self, parent_state, children_states):
        current_location = parent_state[-1] if parent_state else 0
        current_capacity = self.current_capacity(parent_state)
        
        candidates = []
        for child in children_states:
            distance = self.heuristic_greedy_function(current_location, child)
            candidates.append((child, distance))
        
        candidates.sort(key=lambda x: x[1])
        
        if current_capacity >= self.capacity:
            for child, dist in candidates:
                if child in self.dropoff:
                    return child
        
        return candidates[0][0] if candidates else None
    
    def greedy_search(self):
        parent_state = []
        
        while len(parent_state) < self.num_vertices:
            children_states = self.children(parent_state)
            
            if not children_states:
                break
                
            chosen_node = self.choose_node(parent_state, children_states)
            parent_state.append(chosen_node)
        
        return parent_state
    
    def calculate_cost(self, route):
        if not route:
            return float('inf')
        
        cost = self.distance[0][route[0]]
        for i in range(len(route) - 1):
            cost += self.distance[route[i]][route[i + 1]]
        cost += self.distance[route[-1]][0]
        
        return cost


class Local_Search_VRP:
    def __init__(self, num_vertices, capacity, distance):
        self.num_vertices = num_vertices
        self.capacity = capacity
        self.distance = distance
        self.pickup = set(range(1, num_vertices // 2 + 1))
        self.dropoff = set(range(num_vertices // 2 + 1, num_vertices + 1))
        self.iterations = 0  
    
    def calculate_cost(self, route):
        if not route:
            return float('inf')
        
        cost = self.distance[0][route[0]]
        for i in range(len(route) - 1):
            cost += self.distance[route[i]][route[i + 1]]
        cost += self.distance[route[-1]][0]
        
        return cost
    
    def is_valid_route(self, route):
        if len(route) != self.num_vertices:
            return False
        
        current_capacity = 0
        for location in route:
            if location in self.pickup:
                current_capacity += 1
                if current_capacity > self.capacity:
                    return False
            else:
                current_capacity -= 1
        
        visited_pickups = set()
        for location in route:
            if location in self.pickup:
                visited_pickups.add(location)
            else: 
                corresponding_pickup = location - self.num_vertices // 2
                if corresponding_pickup not in visited_pickups:
                    return False
        
        return True
    
    def swap(self, route):
        neighbors = []
        n = len(route)
        
        for i in range(n):
            for j in range(i + 1, n):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                if self.is_valid_route(new_route):
                    neighbors.append(new_route)
        
        return neighbors
    
    def relocate(self, route):
        neighbors = []
        n = len(route)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                new_route = route.copy()
                customer = new_route.pop(i)
                new_route.insert(j, customer)
                
                if self.is_valid_route(new_route):
                    neighbors.append(new_route)
        
        return neighbors
    
    def two_opt(self, route):
        neighbors = []
        n = len(route)
        
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                
                if self.is_valid_route(new_route):
                    neighbors.append(new_route)
        
        return neighbors
    
    def variable_neighborhood_descent(self, initial_route, max_no_improve=10, max_time=280):
        
        self.iterations = 0

        current_route = initial_route.copy()
        current_cost = self.calculate_cost(current_route)

        iterations_without_improvement = 0
        start_time = time.time()

        neighborhoods = [
            ("swap", self.swap),
            ("relocate", self.relocate),
            ("two_opt", self.two_opt),
        ]

        while iterations_without_improvement < max_no_improve:
            if time.time() - start_time > max_time:
                break

            improved = False

            for _name, op in neighborhoods:
                self.iterations += 1
                neighbors = op(current_route)

                best_neighbor = None
                best_neighbor_cost = current_cost

                for neighbor in neighbors:
                    neighbor_cost = self.calculate_cost(neighbor)
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost

                if best_neighbor is not None:
                    current_route = best_neighbor
                    current_cost = best_neighbor_cost
                    improved = True
                    break

            if improved:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

        return current_route, current_cost


def greedy_local_search(num_vertices, capacity, distance, return_iterations=False, verbose=False):

    greedy = Greedy_Search(num_vertices, capacity, distance)
    initial_solution = greedy.greedy_search()
    
    if len(initial_solution) != num_vertices:
        return initial_solution, greedy.calculate_cost(initial_solution) if initial_solution else float('inf')
    
    local_search = Local_Search_VRP(num_vertices, capacity, distance)
    final_solution, final_cost = local_search.variable_neighborhood_descent(initial_solution)

    if verbose:
        print(f"Iterations: {local_search.iterations}")

    if return_iterations:
        return final_solution, final_cost, local_search.iterations
    return final_solution, final_cost


def main():
    import sys
    
    n, k = map(int, input().split())
    
    num_vertices = 2 * n
    
    distance = []
    for _ in range(num_vertices + 1):
        row = list(map(int, input().split()))
        distance.append(row)
    
    start_time = time.time()
    
    solution, cost, iterations = greedy_local_search(num_vertices, k, distance, return_iterations=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(n)
    if solution and len(solution) == num_vertices:
        print(' '.join(map(str, solution)))
    else:
        print("No solution found")
    
    print(f"Cost: {cost}", file=sys.stderr)
    print(f"Iterations: {iterations}", file=sys.stderr)
    print(f"Execution time: {execution_time:.4f} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
