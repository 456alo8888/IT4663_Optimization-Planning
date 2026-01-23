
import random
import math
import time
import argparse
from copy import deepcopy

class CBUSSolver:
    def __init__(self, n, k, distance_matrix):
        self.n = n
        self.k = k
        self.dist = distance_matrix
        self.pickup = list(range(1, n + 1))
        self.dropoff = list(range(n + 1, 2 * n + 1))
        
    def calculate_route_cost(self, route):
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost
    
    def is_valid_route(self, route):

        return 1

    def naive_initialize(self):
        route = [0]
        route.extend(self.pickup)
        route.extend(self.dropoff)
        random.shuffle(route[1:-1])
        route.append(0)
        return route
    
    def greedy_initialize(self):
        route = [0]
        for i in range(1, self.n + 1):
            route.append(i)  
            route.append(i + self.n)  
        route.append(0)
        return route
    
    def nearest_neighbor_initialize(self):
        route = [0]
        unvisited_pickup = set(self.pickup)
        unvisited_dropoff = set(self.dropoff)
        on_bus = set()
        current = 0
        
        while unvisited_pickup or unvisited_dropoff:
            best_next = None
            best_dist = float('inf')
            
            if len(on_bus) < self.k:
                for p in unvisited_pickup:
                    if self.dist[current][p] < best_dist:
                        best_dist = self.dist[current][p]
                        best_next = p
            
            for passenger in on_bus:
                dropoff = passenger + self.n
                if dropoff in unvisited_dropoff:
                    if self.dist[current][dropoff] < best_dist:
                        best_dist = self.dist[current][dropoff]
                        best_next = dropoff
            
            if best_next is None:
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
        new_route = route[:]
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)
        if i != j:
            new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    
    
    def order_crossover(self, parent1, parent2):
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
    
    def simulated_annealing(self, initial_route, coef = 100, max_iter=5000, T0=1000, alpha=0.995):
        current = initial_route[:]
        current_cost = self.calculate_route_cost(current)
        current_violation = self.calculate_route_violation(current)
        current_obj = current_cost + coef * current_violation
        best = current[:]
        best_obj = current_obj
        T = T0
        
        for _ in range(max_iter):
            neighbor = self.swap_mutation(current)
            neighbor_cost = self.calculate_route_cost(neighbor)
            neighbor_violation = self.calculate_route_violation(neighbor)
            neighbor_obj = neighbor_cost + coef * neighbor_violation
            delta = neighbor_obj - current_obj
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_obj = neighbor_obj
                if current_obj < best_obj:
                    best = current[:]
                    best_obj = current_obj
            
            T *= alpha
            if T < 0.1:
                break
        
        return best
    
    def calculate_route_violation(self, route):
        on_bus = set()
        violations = 0
        
        for point in route[1:-1]: 
            if point in self.pickup:
                if len(on_bus) >= self.k:
                    violations += max(len(on_bus) - self.k  + 1, 0)
                on_bus.add(point)
            else:
                passenger = point - self.n
                if passenger not in on_bus:
                    violations += 1
                else:
                    on_bus.remove(passenger)

        violations += len(on_bus) 
        
        return violations

    
    def fitness_function_1(self, route , c = 10):
        cost = self.calculate_route_cost(route)
        violations = self.calculate_route_violation(route)
        return 1.0 / (cost + c * violations)
    
    def fitness_function_2(self, route , c = 00):
        cost = self.calculate_route_cost(route)
        violations = self.calculate_route_violation(route)
        return - (cost + c * violations)
    
    
    
    def genetic_algorithm(self, pop_size=100, generations=200, mutation_rate=0.2, 
                         max_no_improve=50, time_limit=None, start_time=None):
        if self.n > 100:
            pop_size = min(pop_size, 30)
        
        population = []
        population.append(self.greedy_initialize())
        population.append(self.nearest_neighbor_initialize())
        


        
        while len(population) < pop_size:
            route = self.greedy_initialize()
            num_swaps = random.randint(10, 30)  
            for _ in range(num_swaps):
                route = self.swap_mutation(route)
            population.append(route)

        
        # Evaluate fitness
        fitness = [ self.fitness_function_1(r) for r in population] 
    
        
        best_route = max(zip(population, fitness), key=lambda x: x[1])[0]
        best_cost = self.calculate_route_cost(best_route)
        no_improve_count = 0
        
        for gen in range(generations):
            
            if no_improve_count >= max_no_improve:
                break
            
            new_population = []
            
            elite_count = pop_size // 5
            sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            new_population.extend([ind for ind, _ in sorted_pop[:elite_count]])

            if len(new_population) < 2:
                print("Warning: Elite count less than 2, adjusting to 2.")
            
            # Generate offspring
            offspring_count = 0
            max_offspring = pop_size * 2  
            
            while len(new_population) < pop_size and offspring_count < max_offspring:
                offspring_count += 1
                tournament_size = 3 if self.n < 100 else 2
                parent1 = max(random.sample(list(zip(population, fitness)), tournament_size), 
                            key=lambda x: x[1])[0]
                parent2 = max(random.sample(list(zip(population, fitness)), tournament_size), 
                            key=lambda x: x[1])[0]
                
                child = self.order_crossover(parent1, parent2)
                adaptive_mutation_rate = mutation_rate * (1 + no_improve_count / max_no_improve)
                if random.random() < adaptive_mutation_rate:
                    child = self.swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population[:pop_size]  
            fitness = [self.fitness_function_1(r) for r in population]
            current_best = max(zip(population, fitness), key=lambda x: x[1])[0]
            current_cost = self.calculate_route_cost(current_best)
            if current_cost < best_cost:
                best_route = current_best
                best_cost = current_cost
                no_improve_count = 0
            else:
                no_improve_count += 1

            if len(population) == 0:
                print("Population has become empty. Exiting GA.")


        
        return best_route
    
    def solve(self, method='greedy', time_limit=10, **kwargs):
        start_time = time.time()
        
        if method == 'greedy':
            route = self.greedy_initialize()
        elif method == 'nearest':
            route = self.nearest_neighbor_initialize()
        elif method == 'sa':
            initial = self.nearest_neighbor_initialize()
            route = self.simulated_annealing(
                initial,
                coef=kwargs.get('sa_coef', 100),
                max_iter=kwargs.get('sa_max_iter', 5000),
                T0=kwargs.get('sa_T0', 1000),
                alpha=kwargs.get('sa_alpha', 0.995)
            )
        else:
            route = self.genetic_algorithm(
                pop_size=kwargs.get('pop_size', 100),
                generations=kwargs.get('generations', 200),
                mutation_rate=kwargs.get('mutation_rate', 0.2),
                max_no_improve=kwargs.get('max_no_improve', 50),
                time_limit=time_limit,
                start_time=start_time
            )

        end_time = time.time()
        runtime = end_time - start_time
        violations = self.calculate_route_violation(route)
        
        
        return route, runtime, violations


def main():
    line1 = input().split()
    n, k = int(line1[0]), int(line1[1])
    
    distance_matrix = []
    for i in range(2 * n + 1):
        row = list(map(int, input().split()))
        distance_matrix.append(row)
    
    solver = CBUSSolver(n, k, distance_matrix)
    
    best_route = None
    best_cost = float('inf')
    
    if n <= 500: 
        method_pool = ['ga']
    else:
        method_pool = ['ga']

    method_pool =['ga']

    # method_pool = ['sa']
    for method in method_pool:
        try:
            route , runtime, violation = solver.solve(method=method, time_limit=3)
            
            if solver.is_valid_route(route):
                cost = solver.calculate_route_cost(route)
                if cost < best_cost:
                    best_route = route
                    best_cost = cost
                    best_method = method 
        except:
            continue
    
    # Output
    print(n)
    print(' '.join(map(str, best_route[1:-1])))


def read_input_from_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

        line1 = lines[0].strip().split()
        n , k = int(line1[0]) , int(line1[1])
        distance_matrix = []
        for i in range(1, 2*n +2):
            row =list(map(int, lines[i].strip().split())) 
            distance_matrix.append(row)
    return n , k , distance_matrix

def solve_from_file(input_file, output_file, method='ga', **kwargs):
    import sys 
    import os 

    test_name = os.path.splitext(os.path.basename(input_file))[0]

    n , k , distance_matrix = read_input_from_file(input_file)
    solver = CBUSSolver(n , k , distance_matrix= distance_matrix)
    best_route = None 
    best_cost = float('inf')
    best_method = None 

    method_pool = [method] if method else ['ga']

    for m in method_pool:
        try:
            route , runtime, violation = solver.solve(method=m, time_limit=kwargs.get('time_limit', 10), **kwargs)
            
            if solver.is_valid_route(route):
                cost = solver.calculate_route_cost(route)
                if cost < best_cost:
                    best_route = route
                    best_cost = cost
                    best_method = m
        except Exception as e:
            print(f"Error with method {m}: {e}")
            continue
    
    if best_route is None:
        print("No valid route found!")
        return None, float('inf')
    
    print(n)
    print(' '.join(map(str, best_route[1:-1])))
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"n={n}, k={k}\n")
        f.write(f"Method: {best_method}\n")
        if best_method == 'ga':
            f.write(f"  pop_size={kwargs.get('pop_size', 100)}, "
                   f"generations={kwargs.get('generations', 200)}, "
                   f"mutation_rate={kwargs.get('mutation_rate', 0.2)}\n")
        elif best_method == 'sa':
            f.write(f"  T0={kwargs.get('sa_T0', 1000)}, "
                   f"alpha={kwargs.get('sa_alpha', 0.995)}, "
                   f"max_iter={kwargs.get('sa_max_iter', 5000)}, "
                   f"coef={kwargs.get('sa_coef', 100)}\n")
        f.write(f"Best cost: {best_cost}\n")
        f.write(f"Violations: {violation}\n") 
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Route: {' '.join(map(str, best_route[1:-1]))}\n")
    return best_route , best_cost


if __name__ == "__main__":
    import sys 
    import os
    
    parser = argparse.ArgumentParser(description='CBUS Solver with GA and SA')
    parser.add_argument('-i', '--input', type=str, help='Input file path')
    parser.add_argument('-o', '--output', type=str, help='Output file path (optional)')
    parser.add_argument('-m', '--method', type=str, default='ga', 
                       choices=['ga', 'sa', 'greedy', 'nearest'],
                       help='Solution method (default: ga)')
    
    # GA parameters
    parser.add_argument('--pop_size', type=int, default=100, help='GA population size')
    parser.add_argument('--generations', type=int, default=200, help='GA number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='GA mutation rate')
    parser.add_argument('--max_no_improve', type=int, default=50, help='GA early stopping threshold')
    
    # SA parameters
    parser.add_argument('--sa_T0', type=float, default=1000, help='SA initial temperature')
    parser.add_argument('--sa_alpha', type=float, default=0.995, help='SA cooling rate')
    parser.add_argument('--sa_max_iter', type=int, default=5000, help='SA max iterations')
    parser.add_argument('--sa_coef', type=float, default=100, help='SA violation coefficient')
    
    parser.add_argument('--time_limit', type=int, default=10, help='Time limit in seconds')
    
    args = parser.parse_args()
    
    if args.input:
        input_file = args.input
        test_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(output_dir, f"{test_name}_output.txt")
        
        kwargs = {
            'pop_size': args.pop_size,
            'generations': args.generations,
            'mutation_rate': args.mutation_rate,
            'max_no_improve': args.max_no_improve,
            'sa_T0': args.sa_T0,
            'sa_alpha': args.sa_alpha,
            'sa_max_iter': args.sa_max_iter,
            'sa_coef': args.sa_coef,
            'time_limit': args.time_limit
        }
        
        best_route, best_cost = solve_from_file(input_file, output_file, method=args.method, **kwargs)
        print(f"\nBest cost: {best_cost}")
    else:    
        main()