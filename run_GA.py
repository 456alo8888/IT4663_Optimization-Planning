
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
        """Calculate total distance for a route"""
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost
    
    def is_valid_route(self, route):
        """Check if route satisfies capacity and pickup-before-dropoff constraints"""

        return 1
        # on_bus = set()
        
        # for point in route[1:-1]:  # Skip depot at start and end
        #     if point in self.pickup:  # Pickup
        #         if len(on_bus) >= self.k:
        #             return False
        #         on_bus.add(point)
        #     else:  # Dropoff
        #         passenger = point - self.n
        #         if passenger not in on_bus:
        #             return False
        #         on_bus.remove(passenger)
        
        # return len(on_bus) == 0

    def naive_initialize(self):
        """Random permutation initialization"""
        route = [0]
        route.extend(self.pickup)
        route.extend(self.dropoff)
        random.shuffle(route[1:-1])
        route.append(0)
        return route
    
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
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)
        if i != j:
            new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route


    def two_opt_move(self, route):
        """2-opt move: reverse a segment of the route"""
        new_route = route[:]
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)
        
        # Reverse segment between i and j (inclusive)
        new_route[i:j+1] = list(reversed(new_route[i:j+1]))
        return new_route
    
    def insert_move(self, route):
        """Insert move: remove a point and insert it elsewhere"""
        new_route = route[:]
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)
        
        if i != j:
            point = new_route.pop(i)
            new_route.insert(j, point)
        return new_route
    
    def block_swap_move(self, route):
        """Swap two consecutive blocks of random length"""
        new_route = route[:]
        size = len(route) - 2
        
        if size < 4:
            return self.swap_mutation(route)
        
        # Select first block
        block1_start = random.randint(1, size - 2)
        block1_len = random.randint(1, min(3, size - block1_start))
        block1_end = block1_start + block1_len
        
        # Select second block (non-overlapping)
        remaining = list(range(1, block1_start)) + list(range(block1_end, size + 1))
        if not remaining:
            return self.swap_mutation(route)
        
        block2_start = random.choice(remaining)
        block2_len = random.randint(1, min(3, size - block2_start + 1))
        block2_end = min(block2_start + block2_len, size + 1)
        
        # Swap blocks
        block1 = new_route[block1_start:block1_end]
        block2 = new_route[block2_start:block2_end]
        
        if block1_start < block2_start:
            new_route = (new_route[:block1_start] + block2 + 
                        new_route[block1_end:block2_start] + block1 + 
                        new_route[block2_end:])
        else:
            new_route = (new_route[:block2_start] + block1 + 
                        new_route[block2_end:block1_start] + block2 + 
                        new_route[block1_end:])
        
        return new_route
    
    def or_opt_move(self, route):
        """Or-opt move: relocate a sequence of 1-3 consecutive points"""
        new_route = route[:]
        size = len(route) - 2
        
        if size < 2:
            return route
        
        # Select sequence to move
        seq_len = random.randint(1, min(3, size))
        seq_start = random.randint(1, size - seq_len + 1)
        seq_end = seq_start + seq_len
        
        # Select insertion position
        insert_pos = random.randint(1, size + 1)
        while seq_start <= insert_pos < seq_end:
            insert_pos = random.randint(1, size + 1)
        
        # Extract sequence
        sequence = new_route[seq_start:seq_end]
        
        # Remove sequence
        new_route = new_route[:seq_start] + new_route[seq_end:]
        
        # Adjust insert position if needed
        if insert_pos > seq_start:
            insert_pos -= seq_len
        
        # Insert sequence at new position
        new_route = new_route[:insert_pos] + sequence + new_route[insert_pos:]
        
        return new_route
    
    def three_opt_move(self, route):
        """3-opt move: reconnect three segments"""
        new_route = route[:]
        size = len(route) - 2
        
        if size < 3:
            return self.two_opt_move(route)
        
        # Select three cut points
        cuts = sorted(random.sample(range(1, size + 1), 3))
        i, j, k = cuts[0], cuts[1], cuts[2]
        
        # Randomly choose one of the reconnection options
        option = random.randint(0, 3)
        
        if option == 0:
            # Reverse middle segment
            new_route[i:j] = list(reversed(new_route[i:j]))
        elif option == 1:
            # Reverse last segment
            new_route[j:k] = list(reversed(new_route[j:k]))
        elif option == 2:
            # Swap middle and last segments
            new_route = new_route[:i] + new_route[j:k] + new_route[i:j] + new_route[k:]
        else:
            # Reverse and swap
            new_route = (new_route[:i] + list(reversed(new_route[j:k])) + 
                        new_route[i:j] + new_route[k:])
        
        return new_route
    
    def get_neighbor(self, route, move_type=None):
        """Generate neighbor using specified or random move type"""
        if move_type is None:
            # Random selection with weights
            moves = [
                (self.swap_mutation, 0.2),
                (self.two_opt_move, 0.25),
                (self.insert_move, 0.2),
                (self.or_opt_move, 0.15),
                (self.block_swap_move, 0.1),
                (self.three_opt_move, 0.1)
            ]
            
            move_func = random.choices(
                [m[0] for m in moves],
                weights=[m[1] for m in moves]
            )[0]
        else:
            move_func = move_type
        
        return move_func(route)
    
    
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
    
    # def simulated_annealing(self, initial_route, coef = 100, max_iter=5000, T0=1000, alpha=0.995):
    
    #     current = initial_route[:]
    #     current_cost = self.calculate_route_cost(current)
    #     current_violation = self.calculate_route_violation(current)
    #     current_obj = current_cost + coef * current_violation
    #     best = current[:]
    #     best_obj = current_obj
    #     T = T0
        
    #     for epoch in range(max_iter):
    #         # Generate neighbor
    #         print(f"{epoch}")
    #         neighbor = self.get_neighbor(current)
    #         neighbor_cost = self.calculate_route_cost(neighbor)
    #         neighbor_violation = self.calculate_route_violation(neighbor)
    #         neighbor_obj = neighbor_cost + coef * neighbor_violation
    #         # Accept or reject
    #         delta = neighbor_obj - current_obj
    #         if delta < 0 or random.random() < math.exp(-delta / T):
    #             current = neighbor
    #             current_obj = neighbor_obj
    #             if current_obj < best_obj:
    #                 best = current[:]
    #                 best_obj = current_obj
            
    #         T *= alpha
    #         if T < 0.1:
    #             break
        
    #     return best

    def simulated_annealing(self, initial_route, coef=100, max_iter=5000, T0=1000, alpha=0.995):
        """Simulated Annealing with adaptive coefficient and multiple neighborhood operators"""
        # current = initial_route[:]
        
        

        from beam_search import Beam_Search_Graph
        beam = Beam_Search_Graph(2*self.n + 1 , self.dist, self.k)
        beam_sol = beam.beam_search_top_k(
            beam_width= 3 , 
            top_k = 3 , 
            top_neighbors= 3
        )



        if beam_sol:
            current = beam_sol[0][1]
        else:
            current = initial_route[:]

        # Adaptive coefficient: start low, increase gradually
        coef_start = coef * 0.00001  # Start with 10% of target coefficient
        coef_end = coef
        coef_current = coef_start
        
        current_cost = self.calculate_route_cost(current)
        current_violation = self.calculate_route_violation(current)
        current_obj = current_cost + coef_current * current_violation
        
        best = current[:]
        best_obj = current_obj
        best_cost = current_cost
        best_violation = current_violation
        
        T = T0
        no_improve = 0
        accepted = 0

        
        for iteration in range(max_iter):
            # Adaptive coefficient: linear or exponential increase
            # Linear increase
            coef_current = coef_start + (coef_end - coef_start) * (iteration / max_iter)
            
            # Or exponential increase (uncomment to use)
            # progress = iteration / max_iter
            # coef_current = coef_start * ((coef_end / coef_start) ** progress)
            
            # Generate neighbor using random move
            neighbor = self.get_neighbor(current)
            
            neighbor_cost = self.calculate_route_cost(neighbor)
            neighbor_violation = self.calculate_route_violation(neighbor)
            neighbor_obj = neighbor_cost + coef_current * neighbor_violation
            
            # Recalculate current objective with updated coefficient
            current_obj = current_cost + coef_current * current_violation
            
            # Accept or reject
            delta = neighbor_obj - current_obj
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost
                current_violation = neighbor_violation
                current_obj = neighbor_obj
                accepted += 1
                
                # Update best (using final coefficient for fair comparison)
                best_obj_eval = neighbor_cost + coef_end * neighbor_violation
                best_obj_current = best_cost + coef_end * best_violation
                
                if best_obj_eval < best_obj_current:
                    best = neighbor[:]
                    best_cost = neighbor_cost
                    best_violation = neighbor_violation
                    best_obj = best_obj_eval
                    no_improve = 0
                    # print(f"Iter {iteration}: New best = {best_cost}, "
                        #   f"violations = {best_violation}, coef = {coef_current:.2f}")
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # Cooling
            T *= alpha
            
            # Reheating if stuck (with increased coefficient awareness)
            if no_improve > 500:
                T = T0 * 0.3
                no_improve = 0
                # Optionally: increase coefficient faster when stuck
                coef_current = min(coef_current * 1.5, coef_end)
                # print(f"Iter {iteration}: Reheating to T={T:.2f}, coef={coef_current:.2f}")
            
            # Early stopping
            if T < 0.01 or no_improve > 1000:
                print(f"Early stop at iteration {iteration}")
                break
            
            # Progress report every 500 iterations
            if iteration % 500 == 0:
                accept_rate = accepted / (iteration + 1)
                # print(f"Iter {iteration}: T={T:.2f}, coef={coef_current:.2f}, "
                #       f"Best={best_cost} (v={best_violation}), "
                #       f"Current={current_cost} (v={current_violation}), "
                #       f"Accept rate={accept_rate:.3f}")
        
        return best
    
    def calculate_route_violation(self, route):
        """Calculate number of constraint violations in route"""
        on_bus = set()
        violations = 0
        
        for point in route[1:-1]:  # Skip depot
            if point in self.pickup:
                if len(on_bus) >= self.k:
                    violations += max(len(on_bus) - self.k  + 1, 0)
                    # violations += 1
                on_bus.add(point)
            else:
                passenger = point - self.n
                if passenger not in on_bus:
                    violations += 1
                else:
                    on_bus.remove(passenger)

        violations += len(on_bus)  # Remaining passengers not dropped off
        
        return violations
    
    # def finess_function_0(self, route):
    #     """Simple fitness function based on cost"""
    #     cost = self.calculate_route_cost(route)
    #     return 1.0 / cost
    
    def fitness_function_1(self, route , c = 10):
        """Fitness function considering cost and violations"""
        cost = self.calculate_route_cost(route)
        violations = self.calculate_route_violation(route)
        return 1.0 / (cost + c * violations)
    
    def fitness_function_2(self, route , c = 00):
        """Alternative fitness function"""
        cost = self.calculate_route_cost(route)
        violations = self.calculate_route_violation(route)
        return - (cost + c * violations)
    
    
    
    def genetic_algorithm(self, pop_size=100, generations=200, mutation_rate=0.2, 
                         max_no_improve=50, function_choice = 1 , coef = 100,  start_time=None):
        """Genetic Algorithm with early stopping"""
        # Adjust population size based on problem size
        if self.n > 100:
            pop_size = min(pop_size, 30)
        
        # Initialize population
        population = []
        population.append(self.naive_initialize())

        population.append(self.greedy_initialize())
        population.append(self.nearest_neighbor_initialize())

        from beam_search import Beam_Search_Graph
        beam = Beam_Search_Graph(2*self.n + 1 , self.dist, self.k)
        beam_sol = beam.beam_search_top_k(
            beam_width= 5 ,
            top_k = 5 ,
            top_neighbors= 5
        )
        if beam_sol:
            for sol in beam_sol:
                population.append(sol[1])
                if len(population) >= pop_size:
                    break
        

                            
        


        
        while len(population) < pop_size:
            # route = self.greedy_initialize()
            # route = self.naive_initialize()
            # route = self.nearest_neighbor_initialize()
            # pick random from existing and mutate
            base_route = random.choice(population)
            route = base_route[:]
            num_swaps = random.randint(10, 30)  # Fixed range for all sizes
            for _ in range(num_swaps):
                route = self.get_neighbor(route, move_type=self.swap_mutation)
                # route = self.swap_mutation(route)
            population.append(route)

        
        # Evaluate fitness

        if function_choice == 1:
            fitness = [ self.fitness_function_1(r , c= coef) for r in population] #first choice
        else:
            fitness = [ self.fitness_function_2(r , c = coef) for r in population] # second choice
    
        
        best_route = max(zip(population, fitness), key=lambda x: x[1])[0]
        best_cost = self.calculate_route_cost(best_route)
        no_improve_count = 0
        
        for gen in range(generations):
           

            # Early stopping
            if no_improve_count >= max_no_improve:
                break
            
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = pop_size // 5
            sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            new_population.extend([ind for ind, _ in sorted_pop[:elite_count]])

            if len(new_population) < 2:
                print("Warning: Elite count less than 2, adjusting to 2.")
            
            # Generate offspring
            offspring_count = 0
            max_offspring = pop_size * 2  # Limit offspring generation attempts
            
            while len(new_population) < pop_size and offspring_count < max_offspring:
                offspring_count += 1
                # Tournament selection with smaller tournament size for large problems
                tournament_size = 3 if self.n < 100 else 2
                parent1 = max(random.sample(list(zip(population, fitness)), tournament_size), 
                            key=lambda x: x[1])[0]
                parent2 = max(random.sample(list(zip(population, fitness)), tournament_size), 
                            key=lambda x: x[1])[0]
                
                # Crossover
                child = self.order_crossover(parent1, parent2)
                
                # Mutation with adaptive rate
                adaptive_mutation_rate = mutation_rate * (1 + no_improve_count / max_no_improve)
                if random.random() < adaptive_mutation_rate:
                    child = self.swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population[:pop_size]  # Ensure correct size

            if function_choice == 1:
                fitness = [self.fitness_function_1(r , c = coef) for r in population]
            else:
                fitness = [self.fitness_function_2(r , c = coef) for r in population]
            
            # Update best
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
    
    def solve(self, method='greedy', **kwargs):
        """Main solver with multiple methods"""
        start_time = time.time()
        
        if method == 'greedy':
            route = self.greedy_initialize()
        elif method == 'nearest':
            route = self.nearest_neighbor_initialize()
        elif method == 'sa':
            # initial = self.naive_initialize()
            # initial = self.greedy_initialize()
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
                function_choice = kwargs.get('function_choice', 1),
                coef = kwargs.get('ga_coef', 10),
                start_time=start_time
            )
            print("Returned from GA")

        end_time = time.time()
        runtime = end_time - start_time
        violations = self.calculate_route_violation(route)
        
        
        return route, runtime, violations


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
    
    if n <= 500: 
        method_pool = ['ga']
    else:
        method_pool = ['ga']

    for method in method_pool:
        try:
            route , runtime, violation = solver.solve(method=method)
            
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
            route , runtime, violation = solver.solve(method=m, **kwargs)
            
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
                   f"mutation_rate={kwargs.get('mutation_rate', 0.2)}, "
                   f"max_no_improve={kwargs.get('max_no_improve', 50)}, "
                   f"function_choice={kwargs.get('function_choice', 1)}, "
                   f"coef={kwargs.get('ga_coef', 100)}\n")
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
    parser.add_argument('--max_no_improve', type=int, default=1000, help='GA early stopping threshold')
    parser.add_argument('--function_choice', type=float, default=1, help='choose lose function for GA')
    parser.add_argument('--ga_coef', type=float, default=100, help='GA violation coefficient')
    
    # SA parameters
    parser.add_argument('--sa_T0', type=float, default=1000, help='SA initial temperature')
    parser.add_argument('--sa_alpha', type=float, default=0.995, help='SA cooling rate')
    parser.add_argument('--sa_max_iter', type=int, default=5000, help='SA max iterations')
    parser.add_argument('--sa_coef', type=float, default=100, help='SA violation coefficient')
    
    
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
        
        # Prepare kwargs
        kwargs = {
            'pop_size': args.pop_size,
            'generations': args.generations,
            'mutation_rate': args.mutation_rate,
            'max_no_improve': args.max_no_improve,
            'sa_T0': args.sa_T0,
            'sa_alpha': args.sa_alpha,
            'sa_max_iter': args.sa_max_iter,
            'sa_coef': args.sa_coef,
            'function_choice': args.function_choice,
            'ga_coef': args.ga_coef
        }
        
        best_route, best_cost = solve_from_file(input_file, output_file, method=args.method, **kwargs)
        print(f"\nBest cost: {best_cost}")
    else:    
        main()