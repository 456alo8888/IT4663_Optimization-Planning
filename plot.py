import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import math
import copy

class RouteOptimizer:
    def __init__(self, n_cities):
        # Setup dummy data for the simulation
        self.n_cities = n_cities
        # Generate random coordinates for dummy cost calculation
        self.coords = np.random.rand(n_cities, 2) * 100

    def _calculate_distance(self, city1_idx, city2_idx):
        # Dummy distance calculation (Euclidean)
        c1 = self.coords[city1_idx]
        c2 = self.coords[city2_idx]
        return np.sqrt(np.sum((c1 - c2)**2))

    # --- DUMMY IMPLEMENTATIONS OF YOUR EXPECTED METHODS ---
    def calculate_route_cost(self, route):
        """Dummy cost: Sum of distances between consecutive cities"""
        cost = 0
        for i in range(len(route)):
            # Connect to next city, wrapping back to start at the end
            u = route[i]
            v = route[(i + 1) % len(route)]
            cost += self._calculate_distance(u, v)
        return cost

    def calculate_route_violation(self, route):
        """Dummy violation: just return 0 for simplicity in this viz"""
        return 0

    def swap_mutation(self, route):
        """Swaps two random elements in the route."""
        new_route = route[:]
        n = len(new_route)
        if n < 2: return new_route
        i, j = random.sample(range(n), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    # ---------------------------------------------------------


    # NOTE: This is your code, modified slightly to be a GENERATOR (using 'yield')
    # so it can pause and send data to the animation loop.
    def simulated_annealing_generator(self, initial_route, coef=100, max_iter=5000, T0=1000, alpha=0.995):
        """Simulated Annealing instrumented for visualization."""
        current = initial_route[:]
        current_cost = self.calculate_route_cost(current)
        current_violation = self.calculate_route_violation(current)
        current_obj = current_cost + coef * current_violation
        
        best = current[:]
        best_obj = current_obj
        T = T0

        # Yield initial state before loop starts
        yield {
            'iter': 0,
            'T': T,
            'current_route': current,
            'current_obj': current_obj,
            'best_route': best,
            'best_obj': best_obj,
            'status': 'Initial'
        }
        
        for i in range(1, max_iter + 1):
            # Generate neighbor
            neighbor = self.swap_mutation(current)
            neighbor_cost = self.calculate_route_cost(neighbor)
            neighbor_violation = self.calculate_route_violation(neighbor)
            neighbor_obj = neighbor_cost + coef * neighbor_violation
            
            # Accept or reject
            delta = neighbor_obj - current_obj
            status = "Rejected"
            
            # Check acceptance criteria
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_obj = neighbor_obj
                status = "Accepted (Better)" if delta < 0 else "Accepted (Worse)"

                if current_obj < best_obj:
                    best = current[:]
                    best_obj = current_obj
                    status = "New Best Found!"
            
            T *= alpha
            
            # Yield current state for animation frame
            yield {
                'iter': i,
                'T': T,
                'current_route': current,
                'current_obj': current_obj,
                'best_route': best,
                'best_obj': best_obj,
                'status': status
            }
            
            if T < 0.1:
                print(f"Stopping early due to temperature threshold at iteration {i}")
                break
        
        # Yield final state
        yield {
            'iter': i, 'T': T, 'current_route': best, 'current_obj': best_obj,
            'best_route': best, 'best_obj': best_obj, 'status': 'Finished'
        }

# ==========================================
# Visualization Setup
# ==========================================

N_CITIES = 25
optimizer = RouteOptimizer(N_CITIES)

# 1. Create the Naive Initial Solution (a random permutation)
initial_solution = list(range(N_CITIES))
random.shuffle(initial_solution)

# 2. Initialize the SA generator
sa_gen = optimizer.simulated_annealing_generator(
    initial_solution, 
    max_iter=1000, # Reduced iterations for smoother visualization speed
    T0=500, 
    alpha=0.99
)

# 3. Setup Matplotlib Figure Layout
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# Subplot 1: Top row - The Route Blocks
ax_blocks = fig.add_subplot(gs[0, :])
ax_blocks.set_title("Current Route Permutation (Blocks)")
ax_blocks.set_xlabel("Position in Route")
ax_blocks.set_ylabel("City ID")
ax_blocks.set_ylim(0, N_CITIES)
ax_blocks.set_xlim(-0.5, N_CITIES - 0.5)
ax_blocks.set_xticks(range(N_CITIES))

# Initialize bar chart structure
x_pos = np.arange(N_CITIES)
bars = ax_blocks.bar(x_pos, initial_solution, color='skyblue', edgecolor='blue')
best_bars_shadow = ax_blocks.bar(x_pos, initial_solution, color='none', edgecolor='green', alpha=0.5, linestyle='--', linewidth=2, label="Best so far")
ax_blocks.legend()


# Subplot 2: Bottom Left - Objective History
ax_obj = fig.add_subplot(gs[1, 0])
ax_obj.set_title("Objective Function History")
ax_obj.set_xlabel("Iteration")
ax_obj.set_ylabel("Objective Value (Cost)")
line_curr_obj, = ax_obj.plot([], [], 'b-', alpha=0.6, label='Current')
line_best_obj, = ax_obj.plot([], [], 'g-', linewidth=2, label='Best')
ax_obj.legend()
obj_history_curr = []
obj_history_best = []
iter_history = []

# Subplot 3: Bottom Right - Temperature History
ax_temp = fig.add_subplot(gs[1, 1])
ax_temp.set_title("Temperature Schedule")
ax_temp.set_xlabel("Iteration")
ax_temp.set_ylabel("Temperature (T)")
line_temp, = ax_temp.plot([], [], 'r-', label='Temp')
ax_temp.legend()
temp_history = []

# Text annotation for status
status_text = ax_blocks.text(0.02, 0.95, '', transform=ax_blocks.transAxes, 
                             bbox=dict(facecolor='white', alpha=0.8))

def update(frame_data):
    # 1. Update Block Chart (Current Route)
    current_route = frame_data['current_route']
    for bar, height in zip(bars, current_route):
        bar.set_height(height)
        
    # Update shadow of best route
    best_route = frame_data['best_route']
    for bar, height in zip(best_bars_shadow, best_route):
        bar.set_height(height)

    # 2. Update Histories
    iter_history.append(frame_data['iter'])
    obj_history_curr.append(frame_data['current_obj'])
    obj_history_best.append(frame_data['best_obj'])
    temp_history.append(frame_data['T'])

    # Update Line Plots
    line_curr_obj.set_data(iter_history, obj_history_curr)
    line_best_obj.set_data(iter_history, obj_history_best)
    line_temp.set_data(iter_history, temp_history)

    # Rescale axes
    ax_obj.relim()
    ax_obj.autoscale_view()
    ax_temp.relim()
    ax_temp.autoscale_view()

    # Update Status Text
    status_str = (f"Iter: {frame_data['iter']}\n"
                  f"Temp: {frame_data['T']:.2f}\n"
                  f"Curr Obj: {frame_data['current_obj']:.2f}\n"
                  f"Best Obj: {frame_data['best_obj']:.2f}\n"
                  f"Status: {frame_data['status']}")
    status_text.set_text(status_str)

    # Color coding based on status for visual impact
    stat = frame_data['status']
    if "New Best" in stat:
        status_text.set_bbox(dict(facecolor='lightgreen', alpha=0.9))
    elif "Accepted (Worse)" in stat:
        status_text.set_bbox(dict(facecolor='lightyellow', alpha=0.9))
    else:
        status_text.set_bbox(dict(facecolor='white', alpha=0.8))

    return bars, best_bars_shadow, line_curr_obj, line_best_obj, line_temp, status_text

# Create Animation
# interval=20 means 20ms between frames (adjust for speed)
# save_count limits frames if saving to video, important for generators that might run long
ani = animation.FuncAnimation(fig, update, frames=sa_gen, interval=20, repeat=False, save_count=1500)

plt.show()