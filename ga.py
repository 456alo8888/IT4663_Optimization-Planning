import random
import math
import time
import os
from copy import deepcopy
from datetime import datetime

class CBUSSolver:
    def __init__(self, n, k, distance_matrix):
        self.n = n
        self.k = k
        self.dist = distance_matrix
        self.pickup = list(range(1, n + 1))
        self.dropoff = list(range(n + 1, 2 * n + 1))
        
    def calculate_route_cost(self, route):
        """Calculate total distance for a route"""
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost
    
    def is_valid_route(self, route):
        """Check if route satisfies capacity and pickup-before-dropoff constraints"""
        on_bus = set()
        
        for point in route[1:-1]:  # Skip depot at start and end
            if point in self.pickup:  # Pickup
                if len(on_bus) >= self.k:
                    return False
                on_bus.add(point)
            else:  # Dropoff
                passenger = point - self.n
                if passenger not in on_bus:
                    return False
                on_bus.remove(passenger)
        
        return len(on_bus) == 0
    
    def greedy_initialize(self):
        """Greedy initialization: pickup-dropoff pairs in order"""
        route = [0]
        for i in range(1, self.n + 1):
            route.append(i)  # Pickup
            route.append(i + self.n)  # Dropoff
        route.append(0)
        return route
    
    def nearest_neighbor_initialize(self):
        """Nearest neighbor heuristic with capacity constraint"""
        route = [0]
        unvisited_pickup = set(self.pickup)
        unvisited_dropoff = set(self.dropoff)
        on_bus = set()
        current = 0
        
        while unvisited_pickup or unvisited_dropoff:
            best_next = None
            best_dist = float('inf')
            
            # Consider pickups if capacity allows
            if len(on_bus) < self.k:
                for p in unvisited_pickup:
                    if self.dist[current][p] < best_dist:
                        best_dist = self.dist[current][p]
                        best_next = p
            
            # Consider dropoffs for passengers on bus
            for passenger in on_bus:
                dropoff = passenger + self.n
                if dropoff in unvisited_dropoff:
                    if self.dist[current][dropoff] < best_dist:
                        best_dist = self.dist[current][dropoff]
                        best_next = dropoff
            
            if best_next is None:
                # Pick nearest unvisited pickup
                for p in unvisited_pickup:
                    if self.dist[current][p] < best_dist:
                        best_dist = self.dist[current][p]
                        best_next = p
            
            route.append(best_next)
            current = best_next
            
            if best_next in unvisited_pickup:
                unvisited_pickup.remove(best_next)
                on_bus.add(best_next)
            else:
                unvisited_dropoff.remove(best_next)
                on_bus.remove(best_next - self.n)
        
        route.append(0)
        return route
    
    def swap_mutation(self, route):
        """Swap two non-depot points in route"""
        new_route = route[:]
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            i = random.randint(1, len(route) - 2)
            j = random.randint(1, len(route) - 2)
            if i != j:
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if self.is_valid_route(new_route):
                    return new_route
                new_route[i], new_route[j] = new_route[j], new_route[i]
            attempts += 1
        
        return route
    
    def two_opt(self, route, max_iterations=100, time_limit=None, start_time=None):
        """2-opt local search with iteration and time limits"""
        improved = True
        best_route = route[:]
        best_cost = self.calculate_route_cost(best_route)
        iterations = 0
        
        while improved and iterations < max_iterations:
            # Check time limit
            if time_limit and start_time and (time.time() - start_time) > time_limit:
                break
            
            improved = False
            iterations += 1
            
            # Limit combinations for large problems
            route_len = len(best_route)
            step = max(1, route_len // 50)  # Sample every nth position for large routes
            
            for i in range(1, route_len - 2, step):
                for j in range(i + step, route_len - 1, step):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    
                    if self.is_valid_route(new_route):
                        new_cost = self.calculate_route_cost(new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True
                            break
                if improved:
                    break
        
        return best_route
    
    def order_crossover(self, parent1, parent2):
        """Order crossover (OX) preserving route structure"""
        size = len(parent1)
        start, end = sorted(random.sample(range(1, size - 1), 2))
        
        child = [None] * size
        child[0] = 0
        child[-1] = 0
        child[start:end] = parent1[start:end]
        
        fill_pos = end
        for point in parent2[1:-1]:
            if point not in child:
                if fill_pos >= size - 1:
                    fill_pos = 1
                while child[fill_pos] is not None:
                    fill_pos += 1
                    if fill_pos >= size - 1:
                        fill_pos = 1
                child[fill_pos] = point
        
        return child if self.is_valid_route(child) else parent1
    
    def simulated_annealing(self, initial_route, max_iter=5000, T0=1000, alpha=0.995, time_limit=None, start_time=None):
        """Simulated Annealing with time limit"""
        current = initial_route[:]
        current_cost = self.calculate_route_cost(current)
        best = current[:]
        best_cost = current_cost
        T = T0
        
        for iteration in range(max_iter):
            # Check time limit every 100 iterations
            if iteration % 100 == 0 and time_limit and start_time:
                if (time.time() - start_time) > time_limit:
                    break
            
            # Generate neighbor
            neighbor = self.swap_mutation(current)
            neighbor_cost = self.calculate_route_cost(neighbor)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost
            
            T *= alpha
            if T < 0.1:
                break
        
        return best
    
    def genetic_algorithm(self, pop_size=50, generations=200, mutation_rate=0.2):
        """Genetic Algorithm"""
        # Initialize population
        population = []
        population.append(self.greedy_initialize())
        population.append(self.nearest_neighbor_initialize())
        
        while len(population) < pop_size:
            route = self.greedy_initialize()
            for _ in range(random.randint(1, 10)):
                route = self.swap_mutation(route)
            population.append(route)
        
        # Evaluate fitness
        fitness = [1.0 / self.calculate_route_cost(r) for r in population]
        
        best_route = max(zip(population, fitness), key=lambda x: x[1])[0]
        best_cost = self.calculate_route_cost(best_route)
        
        for gen in range(generations):
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = pop_size // 5
            sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            new_population.extend([ind for ind, _ in sorted_pop[:elite_count]])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                parent1 = max(random.sample(list(zip(population, fitness)), 3), key=lambda x: x[1])[0]
                parent2 = max(random.sample(list(zip(population, fitness)), 3), key=lambda x: x[1])[0]
                
                # Crossover
                child = self.order_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self.swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
            fitness = [1.0 / self.calculate_route_cost(r) for r in population]
            
            # Update best
            current_best = max(zip(population, fitness), key=lambda x: x[1])[0]
            current_cost = self.calculate_route_cost(current_best)
            if current_cost < best_cost:
                best_route = current_best
                best_cost = current_cost
        
        return best_route
    
    def solve(self, method='hybrid', time_limit=10):
        """Main solver with multiple methods"""
        start_time = time.time()
        
        # Adjust iterations based on problem size
        two_opt_iters = min(100, max(10, 1000 // self.n))
        sa_iters = min(5000, max(1000, 10000 // self.n))
        
        if method == 'greedy':
            route = self.greedy_initialize()
        elif method == 'nearest':
            route = self.nearest_neighbor_initialize()
        elif method == 'sa':
            initial = self.nearest_neighbor_initialize()
            route = self.simulated_annealing(initial, max_iter=sa_iters, 
                                            time_limit=time_limit, start_time=start_time)
        elif method == 'ga':
            route = self.genetic_algorithm()
        else:  # hybrid
            # Start with nearest neighbor
            route = self.nearest_neighbor_initialize()
            
            # Apply 2-opt
            elapsed = time.time() - start_time
            if elapsed < time_limit * 0.3:
                remaining = time_limit * 0.3 - elapsed
                route = self.two_opt(route, max_iterations=two_opt_iters, 
                                    time_limit=remaining, start_time=time.time())
            
            # Apply SA
            elapsed = time.time() - start_time
            if elapsed < time_limit * 0.7:
                remaining = time_limit * 0.7 - elapsed
                route = self.simulated_annealing(route, max_iter=sa_iters,
                                                time_limit=remaining, start_time=time.time())
            
            # Final 2-opt
            elapsed = time.time() - start_time
            if elapsed < time_limit:
                remaining = time_limit - elapsed
                route = self.two_opt(route, max_iterations=two_opt_iters//2,
                                    time_limit=remaining, start_time=time.time())
        
        return route


def read_input_from_file(file_path):
    """Read input from a text file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    line1 = lines[0].strip().split()
    n, k = int(line1[0]), int(line1[1])
    
    distance_matrix = []
    for i in range(1, 2 * n + 2):
        row = list(map(int, lines[i].strip().split()))
        distance_matrix.append(row)
    
    return n, k, distance_matrix


def log_result_to_file(output_dir, test_name, n, k, route, cost, elapsed_time, method='hybrid'):
    """Log result to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{test_name}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Test Case: {test_name}\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*50 + "\n\n")
        
        f.write(f"Input Parameters:\n")
        f.write(f"  Number of passengers (n): {n}\n")
        f.write(f"  Bus capacity (k): {k}\n")
        f.write(f"  Method: {method}\n\n")
        
        f.write(f"Solution:\n")
        f.write(f"  Total Cost: {cost}\n")
        f.write(f"  Computation Time: {elapsed_time:.4f} seconds\n")
        f.write(f"  Route Length: {len(route)}\n\n")
        
        f.write(f"Route:\n")
        f.write(f"  {' -> '.join(map(str, route))}\n\n")
        
        f.write(f"Route (without depot):\n")
        f.write(f"  {' '.join(map(str, route[1:-1]))}\n")
    
    # print(f"Result logged to: {output_file}")
    return output_file


def solve_from_file(input_file, output_dir):
    """Read input from file, solve, and log result"""
    # Extract test name from file path
    test_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Read input
    # print(f"Reading input from: {input_file}")
    n, k, distance_matrix = read_input_from_file(input_file)
    # print(f"Problem size: n={n}, k={k}")
    
    # Start timing
    start_time = time.time()
    
    # Solve
    solver = CBUSSolver(n, k, distance_matrix)
    
    # Try multiple methods and pick best
    best_route = None
    best_cost = float('inf')
    best_method = None
    
    methods = ['nearest', 'hybrid', 'sa']
    for method in methods:
        try:
            # print(f"Trying method: {method}...")
            route = solver.solve(method=method, time_limit=1)
            if solver.is_valid_route(route):
                cost = solver.calculate_route_cost(route)
                # print(f"  Cost: {cost}")
                if cost < best_cost:
                    best_route = route
                    best_cost = cost
                    best_method = method
        except Exception as e:
            # print(f"  Error: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # Output to console
    # print(f"\nBest solution found:")
    # print(f"  Method: {best_method}")
    # print(f"  Cost: {best_cost}")
    # print(f"  Time: {elapsed_time:.4f}s\n")
    print(n)
    print(' '.join(map(str, best_route[1:-1])))
    
    # Log to file
    log_result_to_file(output_dir, test_name, n, k, best_route, best_cost, elapsed_time, best_method)
    
    return best_route, best_cost


def main():
    # Read input
    line1 = input().split()
    n, k = int(line1[0]), int(line1[1])
    
    distance_matrix = []
    for i in range(2 * n + 1):
        row = list(map(int, input().split()))
        distance_matrix.append(row)
    
    # Solve
    solver = CBUSSolver(n, k, distance_matrix)
    
    # Try multiple methods and pick best
    best_route = None
    best_cost = float('inf')
    
    methods = ['nearest', 'hybrid', 'sa']
    for method in methods:
        try:
            route = solver.solve(method=method, time_limit=3)
            if solver.is_valid_route(route):
                cost = solver.calculate_route_cost(route)
                if cost < best_cost:
                    best_route = route
                    best_cost = cost
        except:
            continue
    
    # Output
    print(n)
    print(' '.join(map(str, best_route[1:-1])))


if __name__ == "__main__":
    import sys
    
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "D:\\HUST\\20251\\IT4663_TULKH\\project\\output"
        solve_from_file(input_file, output_dir)
    else:
        # Default: solve from stdin
        main()