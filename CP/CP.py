from ortools.sat.python import cp_model

import sys
sys.stderr = sys.stdout
model = cp_model.CpModel()

def CP_solver(n, K, C):
    m = 2 * n + 1
    solver = cp_model.CpSolver()
    model = cp_model.CpModel()
    x = {}
    u = {}
    u[0] = 0
    index = {}

    for i in range(m):
        for j in range(m):
            if i != j:
                x[i,j] = model.NewBoolVar(f"x{i}{j}")
    
    for i in range(1, m):
        u[i] = model.NewIntVar(0, K, "load after leaving point " + str(i))

    for i in range(1, m):
        index[i] = model.NewIntVar(1, m - 1, "index visitting point " + str(i))

    model.AddAllDifferent([index[i] for i in range(1, m)])
    for i in range(m):
        model.Add(sum(x[i,j] for j in range(m) if i != j) == 1)
        model.Add(sum(x[j,i] for j in range(m) if i != j) == 1)


    
    M = K + 1 
    for i in range(m):
        for j in range(1, n + 1):
            if i != j:
                model.Add(u[j] == u[i] + 1).OnlyEnforceIf(x[i,j])

    for i in range(1, m):
        for j in range(n + 1, m):
            if i != j:
                model.Add(u[i] == u[j] + 1).OnlyEnforceIf(x[i,j])


    for i in range(1, n + 1):
        model.Add(x[i,0] == 0)
        model.Add(x[0,i + n] == 0)

    for i in range(1, n + 1):
        model.Add(index[i + n] >= index[i] + 1)

    for i in range(1, m):
        for j in range(1, m):
            if i != j:
                model.Add(index[j] == index[i] + 1).OnlyEnforceIf(x[i,j])

    for j in range(1, m):
        model.Add(index[j] == 1).OnlyEnforceIf(x[0, j])

    # last -> depot
    for i in range(1, m):
        model.Add(index[i] == m - 1).OnlyEnforceIf(x[i, 0])

    obj = sum(x[i,j] *  C[i][j] for i in range(m) for j in range(m) if i != j)
    model.Minimize(obj)
    
    
    status = solver.Solve(model)
        
    if status == cp_model.OPTIMAL:
        count = 1
        current_point = 0
        for count in range(1, m):
            i = current_point
            for j in range(m):
                if i != j and solver.Value(x[i,j]) == 1: 

                    print(j, end = " ")
                    current_point = j
                    break

        return solver.ObjectiveValue()
        
    else:
        print(f"No solution found. Status code: {status}")

[n, k] = [int(x) for x in input().split()]
d = []
for i in range(2 * n + 1):
    r = [int(x) for x in input().split()]
    d.append(r)

print(n)
CP_solver(n, k, d)