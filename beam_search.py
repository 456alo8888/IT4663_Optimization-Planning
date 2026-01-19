import copy
import heapq

class Beam_Search_Graph():
    def __init__(self, num_vertices, distance_matrix, capacity):
        self.num_vertices = num_vertices
        self.edges = list()
        self.vertices = [i for i in range(num_vertices)]
        self.num_edges = None
        self.distance_matrix = distance_matrix
        self.num_passengers = (num_vertices-1)//2
        self.capacity = capacity

    '''  
    def update_edges(self):
        for row in range(np.shape(self.distance_matrix)[0]):
            row_edges = list()
            for col in range(np.shape(self.distance_matrix)[1]):
                if row == col:
                    row_edges.append([row, col, 1e9])
                else:
                    row_edges.append([row, col, self.distance_matrix[row][col]])
            self.edges.append(row_edges)
    ''' 
    
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
                
    
    def choose_node(self, frontier):
        idx = 0
        if frontier[0][0] == 0:
            frontier.sort(key= lambda x: x[1])
            parent = frontier[0]
            return parent
        
        else:
            frontier.sort(key= lambda x: x[1])
            if self.current_seat >= self.capacity:
                i = 0
                while frontier[i][0] <= self.num_passengers:
                    i += 1

                parent = frontier[i]
                idx = i
            else:
                parent = frontier[0]
                idx = 0
            return parent
        
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
        """Estimate remaining cost by greedily completing the route.

        Used only for ranking/pruning in beam search. Returns a large number if
        greedy completion fails.
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

        # frontier items: (score, cumulative_cost, last_node, state)
        frontier = [(0.0, 0.0, 0, [0])]

        # min-heap of best complete solutions: (cost, route)
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
                            heapq.heappush(best_complete, (checked_cost, route))
                            if len(best_complete) > top_k:
                                heapq.heappop(best_complete)
                    continue

                children = self.children(last_node, state)
                if not children:
                    continue

                # Consider only the top-N nearest feasible neighbors
                children.sort(key=lambda c: self.distance_matrix[last_node][c])
                children = children[: max(1, int(top_neighbors))]

                for child in children:
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
            frontier = candidates[: max(1, int(beam_width))]

        # best_complete is a min-heap but we trimmed with heappop (removes smallest),
        # so it contains up to top_k solutions but not necessarily ordered.
        results = sorted(best_complete, key=lambda x: x[0])
        return results
        
    def beam_search(self, num_chosen_nodes=3):
        """Backwards-compatible API: return best (cost, route)."""
        results = self.beam_search_top_k(
            beam_width=num_chosen_nodes,
            top_k=1,
            top_neighbors=3,
            use_rollout_heuristic=True,
        )
        if not results:
            return 1e9, None
        return results[0]


def main():
    """
    Input format (from stdin):
        Line 1: n k (number of passengers and capacity)
        Next 2n+1 lines: distance matrix
    
    Output format (to stdout):
        Line 1: n
        Line 2: route (space-separated sequence of points)
    """
    # Read input
    n, k = map(int, input().split())
    num_vertices = 2 * n + 1
    
    distance_matrix = []
    for _ in range(num_vertices):
        row = list(map(int, input().split()))
        distance_matrix.append(row)
    
    # Run beam search (top 3 solutions while expanding top 3 neighbors)
    graph = Beam_Search_Graph(num_vertices, distance_matrix, k)
    top3 = graph.beam_search_top_k(beam_width=3, top_k=3, top_neighbors=3, use_rollout_heuristic=True)

    # Keep stdout compatible: print only the best route
    print(n)
    if top3:
        best_cost, best_route = top3[0]
        print(' '.join(map(str, best_route[1:-1])))
    else:
        print('')

    # Extra info to stderr: show top-3 solutions
    import sys
    for idx, (c, r) in enumerate(top3, start=1):
        print(f"Top {idx} cost: {c}", file=sys.stderr)
        print(f"Top {idx} route: {' '.join(map(str, r[1:-1]))}", file=sys.stderr)


if __name__ == "__main__":
    main()