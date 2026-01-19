

import copy
import random



class Beam_Search_Graph():
    def __init__(self, num_vertices, distance_matrix, capacity):
        self.num_vertices = num_vertices
        self.edges = list()
        self.vertices = [i for i in range(num_vertices)]
        self.num_edges = None
        self.distance_matrix = distance_matrix
        self.num_passengers = (num_vertices-1)//2
        self.capacity = capacity
        
    def compute_path(self, config):
        for i in range(1,len(config)):
            if self.compute_capacity(config[:i]) > self.capacity:
                return 1e9
        else:
            cost = 0
            explore = list()
            for city in range(len(config)-1):
                explore.append(config[city+1])
                cost += self.distance_matrix[config[city]][config[city+1]]
                if config[city+1] > self.num_passengers:
                    if config[city+1] - self.num_passengers not in explore:
                        return 1e9
            return cost
        
    def compute_capacity(self, configuration):
        capp = 0
        for conf in configuration:
            if conf != 0:
                if conf > self.num_passengers:
                    capp -= 1
                else:
                    capp += 1
        return capp
    
    def check_capacity(self, config):
        cap = 0
        for i in range(1,len(config)):
            if config[i] > self.num_passengers:
                cap -= 1
            else:
                cap += 1
            if cap > self.capacity:
                return False
        return True
    
    def check_valid(self, config):
        cap = 0
        for i in range(1,len(config)):
            if config[i] > self.num_passengers:
                cap -= 1
            else:
                cap += 1
            if cap > self.capacity:
                return False
            if config[i] > self.num_passengers:
                if config[i] - self.num_passengers not in config:
                    return False
        return True
        
    def heuristic_greedy_function(self, child, parent):
        return self.distance_matrix[parent][child]
        
    def children(self, node, current_state):
        list_node = self.vertices
        current_seat = self.compute_capacity(current_state)
        res = list()
        for n in list_node:
            if n not in current_state:
                if current_seat >= self.capacity:
                    if n > self.num_passengers:
                        if n - self.num_passengers in current_state:
                            res.append(n)
                else:
                    if n > self.num_passengers:
                        if n - self.num_passengers in current_state:
                            res.append(n)
                        else:
                            continue
                    else:
                        res.append(n)
        return res
        
    def beam_search(self, num_chosen_nodes=3):
        beam_searh_cost = 1e9
        parent_frontier = list()
        parent_frontier.append([0, 0, [0]])
        children_frontier = list()
        res = list()
        
        while True:
            while len(parent_frontier) != 0:
                parent = parent_frontier[0]
                parent_state = parent[2]
                if len(parent_state) == self.num_vertices:
                    res.append(parent_state)
                new_frontier = list()
                for child in self.children(parent[0], parent_state):
                    cost_child = self.heuristic_greedy_function(child, parent[0])
                    child_state = copy.deepcopy(parent_state)
                    child_state.append(child)
                    if self.check_valid(child_state) == True:
                        new_frontier.append([child, cost_child, child_state])
                children_frontier.extend(new_frontier)
                del parent_frontier[0]
            
            if len(children_frontier) == 0:
                break
            children_frontier.sort(key=lambda x: x[1])
            parent_frontier = copy.deepcopy(children_frontier[:num_chosen_nodes])
            children_frontier = list()
        
        for r in res:
            r.append(0)
            cost_search = self.compute_path(r)
            if beam_searh_cost > cost_search:
                beam_searh_cost = cost_search
                opti_path = r
            
        return beam_searh_cost, opti_path




class Local_Search_VRP():
    def __init__(self, num_vertices, distance_matrix, capacity):
        self.num_vertices = num_vertices
        self.distance_matrix = distance_matrix
        self.num_passengers = (num_vertices - 1) // 2
        self.capacity = capacity
    
    def compute_path_cost(self, config):
        cost = 0
        for i in range(len(config) - 1):
            cost += self.distance_matrix[config[i]][config[i + 1]]
        return cost
    
    def is_valid_route(self, config):
        current_capacity = 0
        visited = set()
        
        for i in range(1, len(config) - 1):
            node = config[i]
            visited.add(node)
            
            if node <= self.num_passengers:
                current_capacity += 1
                if current_capacity > self.capacity:
                    return False
            else:
                current_capacity -= 1
                pickup_node = node - self.num_passengers
                if pickup_node not in visited:
                    return False
        
        if len(visited) != 2 * self.num_passengers:
            return False
            
        return True
    
    def two_opt(self, route):
        best_route = route.copy()
        best_cost = self.compute_path_cost(best_route)
        improved = False
        
        n = len(route)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                if self.is_valid_route(new_route):
                    new_cost = self.compute_path_cost(new_route)
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
        
        return best_route, best_cost, improved
    
    def swap(self, route):
        best_route = route.copy()
        best_cost = self.compute_path_cost(best_route)
        improved = False
        
        n = len(route)
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                if self.is_valid_route(new_route):
                    new_cost = self.compute_path_cost(new_route)
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
        
        return best_route, best_cost, improved
    
    def relocate(self, route):
        best_route = route.copy()
        best_cost = self.compute_path_cost(best_route)
        improved = False
        
        n = len(route)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if i == j:
                    continue
                
                new_route = route.copy()
                customer = new_route.pop(i)
                new_route.insert(j, customer)
                
                if self.is_valid_route(new_route):
                    new_cost = self.compute_path_cost(new_route)
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
        
        return best_route, best_cost, improved
    
    def variable_neighborhood_descent(self, initial_route, max_iterations=1000, verbose=False):
        current_route = initial_route.copy()
        current_cost = self.compute_path_cost(current_route)
        
        iteration = 0
        neighborhoods = ['swap', 'relocate', 'two_opt']
        
        if verbose:
            print(f"Initial cost: {current_cost}")
            print(f"Initial route: {current_route}")
        
        while iteration < max_iterations:
            improved = False
            
            for neighborhood in neighborhoods:
                if neighborhood == 'swap':
                    new_route, new_cost, improved = self.swap(current_route)
                elif neighborhood == 'relocate':
                    new_route, new_cost, improved = self.relocate(current_route)
                elif neighborhood == 'two_opt':
                    new_route, new_cost, improved = self.two_opt(current_route)
                
                if improved:
                    current_route = new_route
                    current_cost = new_cost
                    if verbose:
                        print(f"Iteration {iteration}: Improved by {neighborhood} -> Cost: {current_cost}")
                    break
            
            if not improved:
                if verbose:
                    print(f"Local optimum reached at iteration {iteration}")
                break
            
            iteration += 1
        
        if verbose:
            print(f"Final cost: {current_cost}")
            print(f"Final route: {current_route}")
        
        return current_cost, current_route




def hybrid_beam_local_search(num_vertices, distance_matrix, capacity, 
                              beam_width=3, ls_max_iterations=100, verbose=False):
    
    
    if verbose:
        print("="*60)
        print("PHASE 1: BEAM SEARCH (Constructive)")
        print("="*60)
    
    beam = Beam_Search_Graph(num_vertices, distance_matrix, capacity)
    beam_cost, beam_route = beam.beam_search(num_chosen_nodes=beam_width)
    
    if verbose:
        print(f"\n✓ Beam search completed")
        print(f"  Initial cost: {beam_cost}")
        print(f"  Initial route: {beam_route}")
    
        print("\n" + "="*60)
        print("PHASE 2: LOCAL SEARCH (Improvement)")
        print("="*60)
    
    ls = Local_Search_VRP(num_vertices, distance_matrix, capacity)
    final_cost, final_route = ls.variable_neighborhood_descent(
        initial_route=beam_route, 
        max_iterations=ls_max_iterations,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        print(f"Beam search cost:       {beam_cost}")
        print(f"After local search:     {final_cost}")
        improvement = ((beam_cost - final_cost) / beam_cost) * 100
        print(f"Improvement:            {improvement:.2f}%")
        print(f"\nFinal optimized route:  {final_route}")
    
    return final_cost, final_route, beam_cost, beam_route


def main():
    import time
    import sys
    
    # Read input
    n, k = map(int, input().split())
    num_vertices = 2 * n + 1
    
    distance_matrix = []
    for _ in range(num_vertices):
        row = list(map(int, input().split()))
        distance_matrix.append(row)
    
    start_time = time.time()
    
    final_cost, final_route, beam_cost, beam_route = hybrid_beam_local_search(
        num_vertices=num_vertices,
        distance_matrix=distance_matrix,
        capacity=k,
        beam_width=3,
        ls_max_iterations=100
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(n)
    print(' '.join(map(str, final_route[1:-1])))
    
    print(f"Cost: {final_cost}", file=sys.stderr)
    print(f"Execution time: {execution_time:.4f} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
