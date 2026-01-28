import copy
class Greedy_Search_Graph():
    def __init__(self, num_vertices, distance_matrix, capacity):
        self.num_vertices = num_vertices
        self.edges = list()
        self.vertices = [i for i in range(num_vertices)]
        self.num_edges = None
        self.frontier = list()
        self.explored_set = list()
        self.distance_matrix = distance_matrix
        self.visited = [False for i in range(num_vertices)]
        self.cost = [0 for i in range(num_vertices)]
        self.ancestor = [0 for i in range(num_vertices)]
        self.num_passengers = (num_vertices-1)//2
        self.capacity = capacity
        self.current_seat = 0
        
    def children(self, node):
        if node == 0: 
            res = list()
            for n in self.vertices:
                if self.visited[n] == False:
                    if n > self.num_passengers:
                        if self.visited[n - self.num_passengers] == True:
                            res.append(n)
                        else:
                            continue
                    else:
                        res.append(n)
            return res
        
        elif node <= self.num_passengers: 
            res = list()
            for n in self.vertices:
                if self.visited[n] == False:
                    if n > self.num_passengers:
                        if self.visited[n - self.num_passengers] == True:
                            res.append(n)
                        else:
                            continue
                    else:
                        res.append(n)
            return res
            
        else:
            res = list()
            for n in self.vertices:
                if self.visited[n] == False:
                    if n > self.num_passengers: 
                        if self.visited[n - self.num_passengers] == True: 
                            res.append(n)
                        else:
                            continue
                    else:
                        res.append(n)
            return res
    
    def heuristic_greedy_function(self, child, parent):
        return self.distance_matrix[parent][child]
    
    def choose_node(self, frontier):
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
            else:
                parent = frontier[0]

            if parent[0] > self.num_passengers:
                self.current_seat -= 1
            else:
                self.current_seat += 1
            return parent
    
    
    def greedy_search(self):
        greedy_cost = 0
        frontier = list()
        frontier.append([self.vertices[0], 0])
        
        while len(frontier) != 0:
            parent = self.choose_node(frontier)
            greedy_cost += parent[1]
            self.visited[parent[0]] = True
            self.explored_set.append(parent[0])

            new_frontier = list()
            for child in self.children(parent[0]):
                self.ancestor[child] = parent[0]
                self.cost[child] = self.heuristic_greedy_function(child, parent[0])
                new_frontier.append([child, self.cost[child]])
            frontier = copy.deepcopy(new_frontier)

        greedy_cost += self.distance_matrix[parent[0]][0]
        self.explored_set.append(0)
        return greedy_cost, self.explored_set


def main():
    n, k = map(int, input().split())
    num_vertices = 2 * n + 1

    distance_matrix = []
    for _ in range(num_vertices):
        distance_matrix.append(list(map(int, input().split())))

    graph = Greedy_Search_Graph(num_vertices, distance_matrix, k)
    _cost, route = graph.greedy_search()

    print(n)
    print(' '.join(map(str, route[1:-1])))


if __name__ == "__main__":
    main()  
    
