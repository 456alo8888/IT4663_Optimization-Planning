import random
import math
import os

def generate_cbus_test(
    n=5,
    k=2,
    coord_range=(0, 100),
    seed=None
):
    if not (1 <= n <= 1000):
        raise ValueError(f"n must be between 1 and 1000, got {n}")
    if not (1 <= k <= 50):
        raise ValueError(f"k must be between 1 and 50, got {k}")
    if seed is not None:
        random.seed(seed)
    coords = {}
    coords[0] = (
        random.uniform(*coord_range),
        random.uniform(*coord_range)
    )
    for i in range(1, n + 1):
        px = random.uniform(*coord_range)
        py = random.uniform(*coord_range)
        coords[i] = (px, py)
        dx = px + random.uniform(-10, 10)
        dy = py + random.uniform(-10, 10)
        coords[i + n] = (dx, dy)
    size = 2 * n + 1
    dist = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            xi, yi = coords[i]
            xj, yj = coords[j]
            dist[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return n, k, coords, dist

def euclid(a, b):
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))

def build_distance_matrix(coords):
    size = len(coords)
    dist = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            dist[i][j] = euclid(coords[i], coords[j])
    return dist

def generate_cbus_trap(
    n,
    k,
    mode="random",
    coord_range=(0, 100),
    seed=None
):
    if seed is not None:
        random.seed(seed)
    coords = {}
    coords[0] = (50, 50) if mode != "random" else (
        random.uniform(*coord_range),
        random.uniform(*coord_range)
    )
    if mode == "random":
        for i in range(1, n + 1):
            px = random.uniform(*coord_range)
            py = random.uniform(*coord_range)
            coords[i] = (px, py)
            coords[i + n] = (
                px + random.uniform(-10, 10),
                py + random.uniform(-10, 10)
            )
    elif mode == "greedy_trap":
        base_x, base_y = 50, 50
        R = 30
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            coords[i] = (
                base_x + R * math.cos(angle),
                base_y + R * math.sin(angle)
            )
        for i in range(1, n + 1):
            j = (i % n) + 1
            px, py = coords[j]
            coords[i + n] = (px + 2, py + 2)
    elif mode == "cluster_trap":
        cluster_A = (20, 20)
        cluster_B = (180, 180)
        spread = 8
        for i in range(1, n + 1):
            if i <= n // 2:
                px = random.gauss(cluster_A[0], spread)
                py = random.gauss(cluster_A[1], spread)
                dx = random.gauss(cluster_B[0], spread)
                dy = random.gauss(cluster_B[1], spread)
            else:
                px = random.gauss(cluster_B[0], spread)
                py = random.gauss(cluster_B[1], spread)
                dx = random.gauss(cluster_A[0], spread)
                dy = random.gauss(cluster_A[1], spread)
            coords[i] = (px, py)
            coords[i + n] = (dx, dy)
    elif mode == "capacity_trap":
        pickup_center = (20, 20)
        dropoff_center = (180, 180)
        pickup_spread = 5
        dropoff_spread = 5
        for i in range(1, n + 1):
            coords[i] = (
                random.gauss(pickup_center[0], pickup_spread),
                random.gauss(pickup_center[1], pickup_spread)
            )
            coords[i + n] = (
                random.gauss(dropoff_center[0], dropoff_spread),
                random.gauss(dropoff_center[1], dropoff_spread)
            )
    elif mode == "symmetric_trap":
        R = 40
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            coords[i] = (50 + x, 50 + y)
            coords[i + n] = (50 - x, 50 - y)
    elif mode == "star_trap":
        depot_x, depot_y = 50, 50
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            radius = 60 if i % 2 == 0 else 80
            px = depot_x + radius * math.cos(angle)
            py = depot_y + radius * math.sin(angle)
            coords[i] = (px, py)
            dx = depot_x + random.uniform(-8, 8)
            dy = depot_y + random.uniform(-8, 8)
            coords[i + n] = (dx, dy)
    elif mode == "corridor_trap":
        corridor_y = 50
        corridor_width = 5
        for i in range(1, n + 1):
            px = 10 + (180 / n) * i
            py = corridor_y + random.uniform(-corridor_width, corridor_width)
            coords[i] = (px, py)
            dx = 190 - (180 / n) * i
            dy = corridor_y + random.uniform(-corridor_width, corridor_width)
            coords[i + n] = (dx, dy)
    elif mode == "triangle_trap":
        for i in range(1, n + 1):
            px = 20 + random.uniform(-5, 5)
            py = 20 + (i - 1) * (160 / n) + random.uniform(-3, 3)
            coords[i] = (px, py)
            dx = 180 + random.uniform(-5, 5)
            dy = 180 - (i - 1) * (160 / n) + random.uniform(-3, 3)
            coords[i + n] = (dx, dy)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    dist = build_distance_matrix(coords)
    return n, k, coords, dist

def save_test_to_file(filename, n, k, dist):
    with open(filename, 'w') as f:
        f.write(f"{n} {k}\n")
        for i in range(2 * n + 1):
            f.write(' '.join(map(str, dist[i])) + '\n')

def generate_test_suite(output_dir='test_cases'):
    os.makedirs(output_dir, exist_ok=True)
    test_configs = [
        (3, 2, 42, "tiny", "random"),
        (5, 2, 43, "small", "random"),
        (10, 3, 44, "medium_low_capacity", "random"),
        (10, 8, 45, "medium_high_capacity", "random"),
        (20, 5, 46, "medium", "random"),
        (50, 10, 47, "large", "random"),
        (100, 20, 48, "very_large", "random"),
        (200, 30, 49, "huge", "random"),
        (500, 40, 50, "massive", "random"),
        (1000, 50, 51, "extreme", "random"),
    ]
    trap_configs = [
        (10, 5, 100, "greedy_trap_small", "greedy_trap"),
        (20, 8, 101, "greedy_trap_medium", "greedy_trap"),
        (50, 15, 102, "greedy_trap_large", "greedy_trap"),
        (15, 3, 110, "cluster_trap_small_k", "cluster_trap"),
        (30, 8, 111, "cluster_trap_medium", "cluster_trap"),
        (50, 12, 112, "cluster_trap_large", "cluster_trap"),
        (10, 2, 120, "capacity_trap_k2", "capacity_trap"),
        (20, 3, 121, "capacity_trap_k3", "capacity_trap"),
        (30, 5, 122, "capacity_trap_k5", "capacity_trap"),
        (12, 6, 130, "symmetric_trap_even", "symmetric_trap"),
        (20, 8, 131, "symmetric_trap_medium", "symmetric_trap"),
        (30, 10, 132, "symmetric_trap_large", "symmetric_trap"),
        (10, 5, 140, "star_trap_small", "star_trap"),
        (20, 8, 141, "star_trap_medium", "star_trap"),
        (40, 12, 142, "star_trap_large", "star_trap"),
        (15, 5, 150, "corridor_trap_small", "corridor_trap"),
        (25, 8, 151, "corridor_trap_medium", "corridor_trap"),
        (40, 10, 152, "corridor_trap_large", "corridor_trap"),
        (10, 4, 160, "triangle_trap_small", "triangle_trap"),
        (20, 7, 161, "triangle_trap_medium", "triangle_trap"),
        (30, 10, 162, "triangle_trap_large", "triangle_trap"),
    ]
    all_configs = test_configs + trap_configs
    for i, (n, k, seed, desc, mode) in enumerate(all_configs, 1):
        filename = os.path.join(output_dir, f"test{i:02d}_{desc}_n{n}_k{k}.txt")
        print(f"Generating {filename}...")
        if mode == "random":
            n_val, k_val, coords, dist = generate_cbus_test(n=n, k=k, seed=seed)
        else:
            n_val, k_val, coords, dist = generate_cbus_trap(
                n=n, k=k, mode=mode, seed=seed
            )
        save_test_to_file(filename, n_val, k_val, dist)
        print(f"Created: n={n_val}, k={k_val}, mode={mode}, "
              f"matrix size={(2*n_val+1)}x{(2*n_val+1)}")
    print(f"\n{'='*60}")
    print(f"Generated {len(all_configs)} test cases in '{output_dir}/'")
    print(f"  - {len(test_configs)} random tests")
    print(f"  - {len(trap_configs)} adversarial trap tests")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Example: Greedy Trap Test")
    print("="*60)
    n, k, coords, dist = generate_cbus_trap(n=5, k=2, mode="greedy_trap", seed=100)
    print(f"n={n}, k={k}, mode=greedy_trap")
    print("\nCoordinates:")
    print(f"  0 (depot): {coords[0]}")
    for i in range(1, n + 1):
        print(f"  {i} (pickup): {coords[i]}")
    for i in range(n + 1, 2 * n + 1):
        print(f"  {i} (dropoff): {coords[i]}")
    print("\nDistance Matrix (first 5 rows):")
    for i in range(min(5, 2*n + 1)):
        print(f"  Row {i}:", ' '.join(f"{d:4d}" for d in dist[i][:min(10, 2*n+1)]))
    print("\nSaving test to 'sample_trap_test.txt'...")
    save_test_to_file('sample_trap_test.txt', n, k, dist)
    print("\n" + "="*60)
    print("Generating full test suite (random + traps)...")
    print("="*60)
    generate_test_suite()