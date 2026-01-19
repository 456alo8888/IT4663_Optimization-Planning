import copy
import heapq
import random
import time



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

    def _greedy_rollout_extra_cost(self, state, last_node):
        """Heuristic completion cost via greedy rollout.

        Used only for ranking/pruning in beam search.
        Returns a large number if greedy completion fails.
        """
        if len(state) >= self.num_vertices:
            return self.distance_matrix[last_node][0]

        route = state.copy()
        current = last_node
        extra_cost = 0

        while len(route) < self.num_vertices:
            candidates = self.children(current, route)
            if not candidates:
                return 1e9
            next_node = min(candidates, key=lambda c: self.distance_matrix[current][c])
            route.append(next_node)
            extra_cost += self.distance_matrix[current][next_node]
            current = next_node

        extra_cost += self.distance_matrix[current][0]
        return extra_cost

    def beam_search_top_k(
        self,
        beam_width=3,
        top_k=3,
        top_neighbors=3,
        use_rollout_heuristic=True,
    ):
        """Beam search returning the top-K best complete routes.

        - Keeps only `beam_width` best partial states at each depth.
        - Expands only the `top_neighbors` best (nearest) feasible children per state.
        - Returns a list of (cost, route) sorted by cost (route includes depot 0 at both ends).
        """

        beam_width = max(1, int(beam_width))
        top_k = max(1, int(top_k))
        top_neighbors = max(1, int(top_neighbors))

        # frontier items: (score, cumulative_cost, last_node, state)
        frontier = [(0.0, 0.0, 0, [0])]

        # keep best K complete solutions in a max-heap: (-cost, route)
        best_complete = []
        seen_complete = set()

        while frontier:
            candidates = []

            for _score, cumulative_cost, last_node, state in frontier:
                if len(state) == self.num_vertices:
                    route = state + [0]
                    checked_cost = self.compute_path(route)
                    if checked_cost < 1e9:
                        route_key = tuple(route)
                        if route_key not in seen_complete:
                            seen_complete.add(route_key)
                            heapq.heappush(best_complete, (-checked_cost, route))
                            if len(best_complete) > top_k:
                                heapq.heappop(best_complete)
                    continue

                children = self.children(last_node, state)
                if not children:
                    continue

                # Expand only top-N nearest feasible neighbors (neighborhood)
                children.sort(key=lambda c: self.distance_matrix[last_node][c])
                for child in children[:top_neighbors]:
                    edge_cost = self.distance_matrix[last_node][child]
                    child_state = state + [child]
                    if not self.check_valid(child_state):
                        continue

                    child_cum = cumulative_cost + edge_cost
                    if use_rollout_heuristic:
                        score = child_cum + self._greedy_rollout_extra_cost(child_state, child)
                    else:
                        score = child_cum
                    candidates.append((score, child_cum, child, child_state))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            frontier = candidates[:beam_width]

        results = [(-neg_cost, route) for (neg_cost, route) in best_complete]
        results.sort(key=lambda x: x[0])
        return results
        
    def beam_search(self, num_chosen_nodes=3):
        """Backwards-compatible: return best (cost, route)."""
        results = self.beam_search_top_k(
            beam_width=num_chosen_nodes,
            top_k=1,
            top_neighbors=3,
            use_rollout_heuristic=True,
        )
        if not results:
            return 1e9, None
        return results[0]




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
    
    def variable_neighborhood_descent(
        self,
        initial_route,
        max_iterations=100000,
        max_no_improve=10,
        max_time=280,
        verbose=False,
    ):
        """Variable Neighborhood Descent (VND) with time/no-improve stopping.

        This mirrors the stopping behavior used in hill_climbing.py: stop after
        `max_no_improve` consecutive non-improving outer iterations or when `max_time`
        seconds elapse (also capped by `max_iterations`).
        """
        if initial_route is None:
            return 1e9, None, 0

        current_route = initial_route.copy()
        current_cost = self.compute_path_cost(current_route)

        iteration = 0
        iterations_without_improvement = 0
        start_time = time.time()

        neighborhoods = ['swap', 'relocate', 'two_opt']

        if verbose:
            print(f"Initial cost: {current_cost}")
            print(f"Initial route: {current_route}")

        while iteration < max_iterations and iterations_without_improvement < max_no_improve:
            if max_time is not None and (time.time() - start_time) > max_time:
                break

            improved = False
            for neighborhood in neighborhoods:
                if neighborhood == 'swap':
                    new_route, new_cost, neighborhood_improved = self.swap(current_route)
                elif neighborhood == 'relocate':
                    new_route, new_cost, neighborhood_improved = self.relocate(current_route)
                else:  # two_opt
                    new_route, new_cost, neighborhood_improved = self.two_opt(current_route)

                if neighborhood_improved:
                    current_route = new_route
                    current_cost = new_cost
                    improved = True
                    if verbose:
                        print(f"Iteration {iteration}: Improved by {neighborhood} -> Cost: {current_cost}")
                    break

            if improved:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                if verbose:
                    print(f"No improvement ({iterations_without_improvement}/{max_no_improve})")

            iteration += 1

        if verbose:
            print(f"Final cost: {current_cost}")
            print(f"Final route: {current_route}")

        return current_cost, current_route, iteration




def hybrid_beam_local_search(
    num_vertices,
    distance_matrix,
    capacity,
    beam_width=3,
    beam_top_k=3,
    beam_top_neighbors=3,
    ls_max_iterations=100000,
    ls_max_no_improve=10,
    ls_max_time=280,
    verbose=False,
):
    
    
    if verbose:
        print("="*60)
        print("PHASE 1: BEAM SEARCH (Constructive)")
        print("="*60)
    
    beam = Beam_Search_Graph(num_vertices, distance_matrix, capacity)

    # Top-K beam solutions while expanding top-N nearest feasible neighbors.
    beam_solutions = beam.beam_search_top_k(
        beam_width=beam_width,
        top_k=beam_top_k,
        top_neighbors=beam_top_neighbors,
        use_rollout_heuristic=True,
    )

    # Fallback: beam search might fail to produce a complete feasible route for
    # some instances/beam widths. If so, keep the pipeline alive by using a simple
    # deterministic feasible construction.
    if not beam_solutions:
        if verbose:
            print("\n! Beam search found no complete route; falling back to a simple construction")

        num_passengers = (num_vertices - 1) // 2
        route = [0]
        on_bus = []
        remaining_pickups = list(range(1, num_passengers + 1))
        while remaining_pickups or on_bus:
            if len(on_bus) >= capacity or not remaining_pickups:
                p = on_bus.pop(0)
                route.append(p + num_passengers)
            else:
                p = remaining_pickups.pop(0)
                route.append(p)
                on_bus.append(p)
        route.append(0)

        fallback_cost = Local_Search_VRP(num_vertices, distance_matrix, capacity).compute_path_cost(route)
        beam_solutions = [(fallback_cost, route)]

    # Ensure we have unique initial routes
    unique_beam = []
    seen = set()
    for c, r in beam_solutions:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_beam.append((c, r))
    beam_solutions = sorted(unique_beam, key=lambda x: x[0])[: max(1, int(beam_top_k))]
    
    if verbose:
        print(f"\n✓ Beam search completed")
        for idx, (c, r) in enumerate(beam_solutions, start=1):
            print(f"  Beam #{idx} cost:  {c}")
            print(f"  Beam #{idx} route: {r}")
    
        print("\n" + "="*60)
        print("PHASE 2: LOCAL SEARCH (Improvement)")
        print("="*60)
    
    ls = Local_Search_VRP(num_vertices, distance_matrix, capacity)

    # Run local search from each beam-initialized route.
    ls_results = []
    for idx, (init_cost, init_route) in enumerate(beam_solutions, start=1):
        cost, route, iters = ls.variable_neighborhood_descent(
            initial_route=init_route,
            max_iterations=ls_max_iterations,
            max_no_improve=ls_max_no_improve,
            max_time=ls_max_time,
            verbose=verbose,
        )
        ls_results.append(
            {
                'beam_rank': idx,
                'init_cost': init_cost,
                'init_route': init_route,
                'final_cost': cost,
                'final_route': route,
                'ls_iterations': iters,
            }
        )

    ls_results.sort(key=lambda x: x['final_cost'])
    top_ls_results = ls_results[:3]
    best = top_ls_results[0]
    final_cost = best['final_cost']
    final_route = best['final_route']
    ls_iterations = best['ls_iterations']
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        print(f"Best after local search: {final_cost}")
        print(f"Best final route:        {final_route}")

        print("\nTop-3 local-search results:")
        for idx, r in enumerate(top_ls_results, start=1):
            print(f"  LS #{idx}: from Beam #{r['beam_rank']} -> cost={r['final_cost']} iters={r['ls_iterations']}")

    # Backwards-friendly return: best solution + top3 lists
    best_beam_cost, best_beam_route = beam_solutions[0]
    return (
        final_cost,
        final_route,
        best_beam_cost,
        best_beam_route,
        ls_iterations,
        beam_solutions,
        top_ls_results,
    )


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
    
    (
        final_cost,
        final_route,
        beam_cost,
        beam_route,
        ls_iterations,
        beam_solutions,
        top_ls_results,
    ) = hybrid_beam_local_search(
        num_vertices=num_vertices,
        distance_matrix=distance_matrix,
        capacity=k,
        beam_width=3,
        beam_top_k=3,
        beam_top_neighbors=3,
        ls_max_time=280,
        ls_max_no_improve=10
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(n)
    print(' '.join(map(str, final_route[1:-1])))
    
    print(f"Cost: {final_cost}", file=sys.stderr)
    print(f"Local search iterations: {ls_iterations}", file=sys.stderr)
    print(f"Execution time: {execution_time:.4f} seconds", file=sys.stderr)

    # Extra debug info (stderr): top-3 beam + top-3 after local search
    for idx, (c, r) in enumerate(beam_solutions[:3], start=1):
        print(f"Beam Top {idx} cost: {c}", file=sys.stderr)
    for idx, info in enumerate(top_ls_results[:3], start=1):
        print(
            f"LS Top {idx}: from Beam #{info['beam_rank']} cost={info['final_cost']} iters={info['ls_iterations']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
