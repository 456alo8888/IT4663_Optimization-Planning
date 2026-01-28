from ortools.linear_solver import pywraplp
import sys
import time
sec_lst = []

def MIP_solver(n, K, C):
    m = 2 * n + 1

    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {}

    for i in range(m):
        for j in range(m):
            if i != j:
                x[i,j] = solver.BoolVar("x(" + str(i) + str(j) + ")")


    f = [[[solver.IntVar(0, 1, "f(" + str(p) + "," + str(i) + "," + str(j) + ")") for j in range(m)]
          for i in range(m)]
         for p in range(n)]

    for p in range(n):
        for i in range(m):
            solver.Add(f[p][i][i] == 0)
            for j in range(m):
                if i != j:
                    solver.Add(f[p][i][j] <= x[i,j])

    for i in range(m):
        solver.Add(sum(x[i,j] for j in range(m) if j != i) == 1)
        solver.Add(sum(x[j,i] for j in range(m) if j != i) == 1)

    for i in range(m):
        for j in range(m):
            if i != j:
                solver.Add(sum(f[p][i][j] for p in range(n)) <= K * x[i,j])

    for p in range(n):
        for v in range(1, m):
            solver.Add(f[p][0][v] == 0)
            solver.Add(f[p][v][0] == 0)

    for p in range(n):
        pickup = p + 1
        drop = p + n + 1
        for v in range(m):
            out_v = sum(f[p][v][j] for j in range(m) if j != v)
            in_v  = sum(f[p][i][v] for i in range(m) if i != v)

            if v == pickup:
                solver.Add(out_v - in_v == 1)
            elif v == drop:
                solver.Add(out_v - in_v == -1)
            else:
                solver.Add(out_v - in_v == 0)

    for sec in sec_lst:
        solver.Add(
            sum(x[i,j] for i in sec for j in sec if i != j) <= len(sec) - 1
        )

    
    solver.Minimize(sum(C[i][j] * x[i,j] for i in range(m) for j in range(m) if i != j))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # print(time.time() - intial)
        next_node = [-1] * m
        for i in range(m):
            for j in range(m):
                if i != j and x[i,j].solution_value() == 1:  
                    next_node[i] = j
                    break

        # Extract all cycles
        seen = [0] * m
        cycles = []
        for start in range(m):
            if seen[start]:
                continue
            cur = start
            cycle = []
            while not seen[cur]:
                seen[cur] = True
                cycle.append(cur)
                cur = next_node[cur]
            cycles.append(cycle)

        
        subtours = [c for c in cycles if len(c) < m]
        if subtours:
            sec_lst.extend(subtours)
            return None
        
        for u in cycles[0][1:]:
            print(u, end=" ")
        print()
        return solver.Objective().Value()
    else:
        return None


n, K = map(int, input().split())
m = 2 * n + 1
C = [list(map(int, input().split())) for _ in range(m)]

print(n)
intial = time.time()
while True:
    result = MIP_solver(n, K, C)
    if result is not None:
       
        break
    
