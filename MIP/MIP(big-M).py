from ortools.linear_solver import pywraplp
import time

initial = time.time()
def MIP_solver(n, capacity, dis):
    model = pywraplp.Solver.CreateSolver("SCIP")
    x = {}
    u = {}
    u[0] = 0
    index = {}
    m = 2 * n + 1
    
    for i in range(m):
        for j in range(m):
            if i != j:
                x[i,j] = model.BoolVar("x(" + str(i) + str(j) + ")") 
    
    for i in range(1, n + 1):
        u[i] = model.IntVar(1, capacity, "load after leaving point " + str(i))

    for i in range(n + 1, m):
        u[i] = model.IntVar(0, capacity-1, "load after leaving point " + str(i))
    for i in range(1, 2*n + 1):
        index[i] = model.IntVar(1, 2*n + 1, "index visitting point " + str(i))

    for i in range(m):
        model.Add(sum(x[i,j] for j in range(m) if i != j) == 1)
        model.Add(sum(x[j,i] for j in range(m) if i != j) == 1)

    
    M = capacity + 1 
    for i in range(m):
        for j in range(1, n + 1):
            if i != j:
                model.Add(u[j] - u[i] - 1 >= -M * (1 - x[i,j]))
                model.Add(u[j] - u[i] - 1 <= M * (1 - x[i,j]))

    for i in range(1, m):
        for j in range(n + 1, m):
            if i != j:
                model.Add(u[i] - u[j] - 1 >= -M * (1 - x[i,j]))
                model.Add(u[i] - u[j] - 1 <= M * (1 - x[i,j]))

    for i in range(1, n + 1):
        model.Add(x[i,0] == 0)
        model.Add(x[0,i + n] == 0)

    for i in range(1, n + 1):
        model.Add(index[i + n] >= index[i] + 1)

    
    M_index = m 
    for i in range(1, m):
        for j in range(1, m):
            if i != j:
            
                model.Add(index[j] - index[i] - 1 >= -M_index * (1 - x[i,j]))
                model.Add(index[j] - index[i] - 1 <= M_index * (1 - x[i,j]))

    obj = sum(x[i,j] *  dis[i][j] for i in range(m) for j in range(m) if i != j)
    model.Minimize(obj)
    
    model.SetTimeLimit(300000)  
    status = model.Solve()
    
    
    if status == pywraplp.Solver.OPTIMAL:
        count = 1
        current_point = 0
        for count in range(1, m):
            i = current_point
            for j in range(m):
                if i != j and x[i,j].solution_value() == 1:
                    print(j, end = " ")
                    current_point = j
                    break
            
        print()
    else:
        print(f"No solution found. Status code: {status}")

[n, k] = [int(x) for x in input().split()]
d = []


for i in range(2 * n + 1):
    r = [int(x) for x in input().split()]
    d.append(r)

print(n)
MIP_solver(n, k, d)

