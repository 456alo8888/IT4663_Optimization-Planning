import time

import numpy as np


class CBUSSolverReduced:
    def __init__(self, dist_matrix, n_customers, capacity):
        self.dist_matrix = np.array(dist_matrix, dtype=float)
        self.num_points = len(dist_matrix)
        self.n_customers = n_customers
        self.capacity = capacity

        np.fill_diagonal(self.dist_matrix, float("inf"))

        self.best_path = []
        self.min_cost = float("inf")

    def reduce_matrix(self, matrix):
        reduction_cost = 0
        temp_matrix = matrix.copy()

        for i in range(len(temp_matrix)):
            row_min = np.min(temp_matrix[i])
            if row_min != float("inf") and row_min > 0:
                reduction_cost += row_min
                temp_matrix[i] -= row_min

        for j in range(len(temp_matrix)):
            col_min = np.min(temp_matrix[:, j])
            if col_min != float("inf") and col_min > 0:
                reduction_cost += col_min
                temp_matrix[:, j] -= col_min

        return temp_matrix, reduction_cost

    def solve(self):
        start = time.perf_counter()
        initial_matrix, initial_bound = self.reduce_matrix(self.dist_matrix)

        visited = [False] * self.num_points
        visited[0] = True

        self._backtrack(initial_matrix, 0, initial_bound, [0], 0, visited)

        end = time.perf_counter()
        return self.best_path, self.min_cost, end - start

    def _backtrack(self, matrix, curr_node, current_bound, path, load, visited):
        if len(path) == self.num_points:
            final_cost = current_bound + matrix[curr_node][0]
            if final_cost < self.min_cost:
                self.min_cost = final_cost
                self.best_path = path + [0]
            return

        next_points = []
        for next_node in range(1, self.num_points):
            if not visited[next_node] and self._is_valid(next_node, load, visited):
                next_points.append(next_node)

        next_points.sort(key=lambda x: matrix[curr_node][x])

        for next_node in next_points:
            new_matrix = matrix.copy()

            new_matrix[curr_node, :] = float("inf")
            new_matrix[:, next_node] = float("inf")
            new_matrix[next_node, 0] = (
                float("inf")
                if len(path) < self.num_points - 1
                else new_matrix[next_node, 0]
            )

            new_matrix, reduction_cost = self.reduce_matrix(new_matrix)
            new_bound = current_bound + matrix[curr_node][next_node] + reduction_cost

            if new_bound < self.min_cost:
                visited[next_node] = True
                new_load = load + (1 if next_node <= self.n_customers else -1)

                self._backtrack(
                    new_matrix,
                    next_node,
                    new_bound,
                    path + [next_node],
                    new_load,
                    visited,
                )

                visited[next_node] = False

    def _is_valid(self, next_node, load, visited):
        if 1 <= next_node <= self.n_customers:
            return load < self.capacity
        else:
            pickup_node = next_node - self.n_customers
            return visited[pickup_node]


if __name__ == "__main__":
    n_customers, capacity = map(int, input().split())
    dist_matrix = []
    for _ in range(2 * n_customers + 1):
        row = list(map(int, input().split()))
        dist_matrix.append(row)
    solver = CBUSSolverReduced(dist_matrix, n_customers, capacity)
    best_path, min_cost, exec_time = solver.solve()
    print(n_customers)
    print(" ".join(map(str, best_path[1:-1])))
