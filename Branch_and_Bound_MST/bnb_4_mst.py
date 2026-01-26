"""Branch and Bound with MST lower bound for Capacitated Pickup and Delivery Problem."""

import heapq
import time


def prim_mst(matrix, nodes):
    if not nodes:
        return 0

    pq = [(0, nodes[0])]
    visited = set()
    total_weight = 0

    while pq and len(visited) < len(nodes):
        weight, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)
        total_weight += weight

        for v in nodes:
            if v not in visited:
                heapq.heappush(pq, (matrix[u][v], v))

    return total_weight


def hustack(dist_matrix, n_customer, capacity, mode):
    solver = CBUSSolver(dist_matrix, n_customer, capacity, mode)
    best_path, min_cost, run_time = solver.solve()
    print(n_customer)
    print(" ".join(map(str, best_path[1:-1])))


class CBUSSolver:
    def __init__(self, dist_matrix, n_customers, capacity, mode):
        self.dist_matrix = dist_matrix
        self.num_points = len(dist_matrix)
        self.n_customers = n_customers
        self.capacity = capacity
        self.visited = [False] * self.num_points
        self.current_path = [0]
        self.current_cost = 0
        self.current_load = 0
        self.best_path = []
        self.min_cost = float("inf")
        # bound mode: 0 - backtrack
        # 1 - Min edge bound
        # 2 - Outgoing edge bound
        # 3 - style bound
        # 4 - mst
        self.bound_mode = mode
        # Mode 1
        self.c_min = min(
            dist_matrix[i][j]
            for i in range(self.num_points)
            for j in range(self.num_points)
            if i != j
        )
        # Mode 2
        self.min_out = []
        for i in range(self.num_points):
            edges = [dist_matrix[i][j] for j in range(self.num_points) if i != j]
            self.min_out.append(min(edges))
        # Mode 3
        self.min_in = []
        for j in range(self.num_points):
            edges = [dist_matrix[i][j] for i in range(self.num_points) if i != j]
            self.min_in.append(min(edges))

    def solve(self):
        start = time.perf_counter()
        self.visited[0] = True
        self._backtrack()
        end = time.perf_counter()
        return self.best_path, self.min_cost, end - start

    def _is_valid_move(self, next_point):
        if self.visited[next_point]:
            return False
        elif 1 <= next_point <= self.n_customers:
            return self.current_load < self.capacity
        else:
            pickup_point = next_point - self.n_customers
            return self.visited[pickup_point]

    def _backtrack(self):
        if len(self.current_path) == self.num_points:
            total_cost = self.current_cost + self.dist_matrix[self.current_path[-1]][0]
            if total_cost < self.min_cost:
                self.min_cost = total_cost
                self.best_path = list(self.current_path) + [0]
            return

        good_points = []
        for point in range(1, self.num_points):
            if self._is_valid_move(point):
                good_points.append(point)
        good_points.sort(key=lambda x: self.dist_matrix[self.current_path[-1]][x])

        for next_point in good_points:
            if self._is_valid_move(next_point):
                load_change = 0

                if 1 <= next_point <= self.n_customers:
                    load_change = 1
                else:
                    load_change = -1

                new_cost = (
                    self.current_cost
                    + self.dist_matrix[self.current_path[-1]][next_point]
                )
                # Mode 0
                lb = new_cost
                # Mode 1
                if self.bound_mode == 1:
                    remaining_steps = (self.num_points - len(self.current_path)) + 1
                    lb = new_cost + remaining_steps * self.c_min
                # Mode 2
                if self.bound_mode == 2:
                    lb = new_cost
                    for p in range(1, self.num_points):
                        if not self.visited[p]:
                            lb += self.min_out[p]
                # mode 3
                if self.bound_mode == 3:
                    lb = new_cost
                    num_unvisited = 0
                    for p in range(1, self.num_points):
                        if not self.visited[p] and p != next_point:
                            lb += (self.min_in[p] + self.min_out[p]) / 2
                            num_unvisited += 1
                    if num_unvisited:
                        lb += (self.min_out[next_point] + self.min_in[0]) / 2
                    else:
                        lb += self.dist_matrix[next_point][0]
                # mode 4
                if self.bound_mode == 4:
                    remaining_nodes = [
                        i
                        for i in range(1, self.num_points)
                        if not self.visited[i] and i != next_point
                    ]

                    if not remaining_nodes:
                        lb = new_cost + self.dist_matrix[next_point][0]
                    else:
                        lb = (
                            new_cost
                            + prim_mst(self.dist_matrix, remaining_nodes)
                            + self.min_out[next_point]
                            + self.min_in[0]
                        )
                #####
                if lb < self.min_cost:
                    self.visited[next_point] = True
                    self.current_path.append(next_point)
                    self.current_cost = new_cost
                    self.current_load += load_change

                    self._backtrack()

                    self.current_load -= load_change
                    self.current_cost -= self.dist_matrix[self.current_path[-2]][
                        next_point
                    ]
                    self.current_path.pop()
                    self.visited[next_point] = False


if __name__ == "__main__":
    n_customers, capacity = map(int, input().split())
    dist_matrix = []
    for _ in range(2 * n_customers + 1):
        row = list(map(int, input().split()))
        dist_matrix.append(row)
    hustack(dist_matrix, n_customers, capacity, mode=4)
