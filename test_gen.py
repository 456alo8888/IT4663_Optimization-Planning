# import random
# import math
# import os

# def generate_cbus_test(
#     n=5,
#     k=2,
#     coord_range=(0, 100),
#     seed=None
# ):
#     """
#     Generate a CBUS test case
#     Constraints: 1 ≤ n ≤ 1000, 1 ≤ k ≤ 50
#     Returns:
#         n, k, coords, dist_matrix
#     """
#     # Validate constraints
#     if not (1 <= n <= 1000):
#         raise ValueError(f"n must be between 1 and 1000, got {n}")
#     if not (1 <= k <= 50):
#         raise ValueError(f"k must be between 1 and 50, got {k}")
    
#     if seed is not None:
#         random.seed(seed)

#     coords = {}

#     # Depot at point 0
#     coords[0] = (
#         random.uniform(*coord_range),
#         random.uniform(*coord_range)
#     )

#     # Pickups (1 to n) and dropoffs (n+1 to 2n)
#     for i in range(1, n + 1):
#         # Pickup point i
#         px = random.uniform(*coord_range)
#         py = random.uniform(*coord_range)
#         coords[i] = (px, py)

#         # Dropoff point i+n (near pickup)
#         dx = px + random.uniform(-10, 10)
#         dy = py + random.uniform(-10, 10)
#         coords[i + n] = (dx, dy)

#     # Distance matrix (2n+1) x (2n+1)
#     # Points: 0 (depot), 1..n (pickups), n+1..2n (dropoffs)
#     size = 2 * n + 1
#     dist = [[0] * size for _ in range(size)]

#     for i in range(size):
#         for j in range(size):
#             xi, yi = coords[i]
#             xj, yj = coords[j]
#             # Integer distances as per problem format
#             dist[i][j] = int(round(math.hypot(xi - xj, yi - yj)))

#     return n, k, coords, dist


# def save_test_to_file(filename, n, k, dist):
#     """Save test case in the required format"""
#     with open(filename, 'w') as f:
#         # Line 1: n and k
#         f.write(f"{n} {k}\n")
        
#         # Next 2n+1 lines: distance matrix
#         for i in range(2 * n + 1):
#             f.write(' '.join(map(str, dist[i])) + '\n')


# def generate_test_suite(output_dir='test_cases'):
#     """Generate a suite of test cases with varying difficulty"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     test_configs = [
#         # (n, k, seed, description)
#         (3, 2, 42, "tiny"),
#         (5, 2, 43, "small"),
#         (10, 3, 44, "medium_low_capacity"),
#         (10, 8, 45, "medium_high_capacity"),
#         (20, 5, 46, "medium"),
#         (50, 10, 47, "large"),
#         (100, 20, 48, "very_large"),
#         (200, 30, 49, "huge"),
#         (500, 40, 50, "massive"),
#         (1000, 50, 51, "extreme"),
#     ]
    
#     for i, (n, k, seed, desc) in enumerate(test_configs, 1):
#         filename = os.path.join(output_dir, f"test{i:02d}_{desc}_n{n}_k{k}.txt")
#         print(f"Generating {filename}...")
        
#         n_val, k_val, coords, dist = generate_cbus_test(n=n, k=k, seed=seed)
#         save_test_to_file(filename, n_val, k_val, dist)
        
#         print(f"  ✓ Created: n={n_val}, k={k_val}, matrix size={(2*n_val+1)}x{(2*n_val+1)}")
    
#     print(f"\nGenerated {len(test_configs)} test cases in '{output_dir}/'")


# if __name__ == "__main__":
#     # Example: Generate single test
#     n, k, coords, dist = generate_cbus_test(n=3, k=2, seed=42)
    
#     print(f"n={n}, k={k}")
#     print("\nCoordinates:")
#     print("  0 (depot):", coords[0])
#     for i in range(1, n + 1):
#         print(f"  {i} (pickup): {coords[i]}")
#     for i in range(n + 1, 2 * n + 1):
#         print(f"  {i} (dropoff): {coords[i]}")
    
#     print("\nDistance Matrix (first 5 rows):")
#     for i in range(min(5, 2*n + 1)):
#         print(f"  Row {i}:", ' '.join(f"{d:4d}" for d in dist[i][:min(10, 2*n+1)]))
    
#     # Save single test
#     print("\nSaving test to 'sample_test.txt'...")
#     save_test_to_file('sample_test.txt', n, k, dist)
    
#     # Generate full test suite
#     print("\n" + "="*60)
#     print("Generating full test suite...")
#     print("="*60)
#     generate_test_suite()


import random
import math
import os

def generate_cbus_test(
    n=5,
    k=2,
    coord_range=(0, 100),
    seed=None
):
    """
    Generate a CBUS test case
    Constraints: 1 ≤ n ≤ 1000, 1 ≤ k ≤ 50
    Returns:
        n, k, coords, dist_matrix
    """
    # Validate constraints
    if not (1 <= n <= 1000):
        raise ValueError(f"n must be between 1 and 1000, got {n}")
    if not (1 <= k <= 50):
        raise ValueError(f"k must be between 1 and 50, got {k}")
    
    if seed is not None:
        random.seed(seed)

    coords = {}

    # Depot at point 0
    coords[0] = (
        random.uniform(*coord_range),
        random.uniform(*coord_range)
    )

    # Pickups (1 to n) and dropoffs (n+1 to 2n)
    for i in range(1, n + 1):
        # Pickup point i
        px = random.uniform(*coord_range)
        py = random.uniform(*coord_range)
        coords[i] = (px, py)

        # Dropoff point i+n (near pickup)
        dx = px + random.uniform(-10, 10)
        dy = py + random.uniform(-10, 10)
        coords[i + n] = (dx, dy)

    # Distance matrix (2n+1) x (2n+1)
    # Points: 0 (depot), 1..n (pickups), n+1..2n (dropoffs)
    size = 2 * n + 1
    dist = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            xi, yi = coords[i]
            xj, yj = coords[j]
            # Integer distances as per problem format
            dist[i][j] = int(round(math.hypot(xi - xj, yi - yj)))

    return n, k, coords, dist


def euclid(a, b):
    """Calculate Euclidean distance between two points"""
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))


def build_distance_matrix(coords):
    """Build distance matrix from coordinates"""
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
    """
    Generate adversarial test cases to trap algorithms
    
    Modes:
    - random: baseline random distribution
    - greedy_trap: pickup close to OTHER's dropoff (traps greedy)
    - cluster_trap: two distant clusters, pickups/dropoffs separated
    - capacity_trap: small k, pickups close, dropoffs far (capacity violation)
    - symmetric_trap: strong symmetry causing branch explosion
    - star_trap: star pattern from depot (traps nearest neighbor)
    - corridor_trap: narrow corridor with optimal path non-obvious
    - triangle_trap: triangular inequality trap
    """
    if seed is not None:
        random.seed(seed)

    coords = {}

    # Depot at origin for most trap modes
    coords[0] = (50, 50) if mode != "random" else (
        random.uniform(*coord_range),
        random.uniform(*coord_range)
    )

    # --------------------------------------------------
    # MODE 1: RANDOM (baseline)
    # --------------------------------------------------
    if mode == "random":
        for i in range(1, n + 1):
            px = random.uniform(*coord_range)
            py = random.uniform(*coord_range)
            coords[i] = (px, py)
            coords[i + n] = (
                px + random.uniform(-10, 10),
                py + random.uniform(-10, 10)
            )

    # --------------------------------------------------
    # MODE 2: GREEDY TRAP
    # Pickup gần dropoff của NGƯỜI KHÁC
    # → Greedy chọn sai → vi phạm ràng buộc
    # --------------------------------------------------
    elif mode == "greedy_trap":
        base_x, base_y = 50, 50
        R = 30

        # Pickup on circle
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            coords[i] = (
                base_x + R * math.cos(angle),
                base_y + R * math.sin(angle)
            )

        # Dropoff rotated (i -> i+1 mod n)
        # Pickup i gần dropoff (i+1), traps greedy!
        for i in range(1, n + 1):
            j = (i % n) + 1  # Next pickup
            px, py = coords[j]
            coords[i + n] = (px + 2, py + 2)

    # --------------------------------------------------
    # MODE 3: CLUSTER TRAP
    # Hai cụm xa nhau, pickup và dropoff ở khác cụm
    # → Phải qua lại giữa 2 cụm nhiều lần
    # --------------------------------------------------
    elif mode == "cluster_trap":
        cluster_A = (20, 20)
        cluster_B = (180, 180)
        spread = 8

        for i in range(1, n + 1):
            if i <= n // 2:
                # Pickup in cluster A, dropoff in cluster B
                px = random.gauss(cluster_A[0], spread)
                py = random.gauss(cluster_A[1], spread)
                dx = random.gauss(cluster_B[0], spread)
                dy = random.gauss(cluster_B[1], spread)
            else:
                # Pickup in cluster B, dropoff in cluster A
                px = random.gauss(cluster_B[0], spread)
                py = random.gauss(cluster_B[1], spread)
                dx = random.gauss(cluster_A[0], spread)
                dy = random.gauss(cluster_A[1], spread)

            coords[i] = (px, py)
            coords[i + n] = (dx, dy)

    # --------------------------------------------------
    # MODE 4: CAPACITY TRAP
    # k nhỏ, pickup gần nhau, dropoff RẤT xa
    # → Không thể pick nhiều người trước khi drop
    # --------------------------------------------------
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

    # --------------------------------------------------
    # MODE 5: SYMMETRIC TRAP
    # Đối xứng mạnh → nhiều nghiệm có cost giống nhau
    # → Branch explosion, khó chọn nghiệm tốt
    # --------------------------------------------------
    elif mode == "symmetric_trap":
        R = 40
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            coords[i] = (50 + x, 50 + y)
            coords[i + n] = (50 - x, 50 - y)  # Symmetric dropoff

    # --------------------------------------------------
    # MODE 6: STAR TRAP
    # Pickup phân tán đều từ depot (star pattern)
    # Dropoff gần depot
    # → Nearest neighbor từ depot trap (chọn xa trước)
    # --------------------------------------------------
    elif mode == "star_trap":
        depot_x, depot_y = 50, 50
        
        for i in range(1, n + 1):
            angle = 2 * math.pi * (i - 1) / n
            # Pickup far from depot (alternating distances)
            radius = 60 if i % 2 == 0 else 80
            px = depot_x + radius * math.cos(angle)
            py = depot_y + radius * math.sin(angle)
            coords[i] = (px, py)
            
            # Dropoff close to depot
            dx = depot_x + random.uniform(-8, 8)
            dy = depot_y + random.uniform(-8, 8)
            coords[i + n] = (dx, dy)

    # --------------------------------------------------
    # MODE 7: CORRIDOR TRAP
    # Các điểm nằm trên 1 corridor hẹp
    # Optimal path không rõ ràng
    # --------------------------------------------------
    elif mode == "corridor_trap":
        corridor_y = 50
        corridor_width = 5
        
        for i in range(1, n + 1):
            # Pickup along corridor
            px = 10 + (180 / n) * i
            py = corridor_y + random.uniform(-corridor_width, corridor_width)
            coords[i] = (px, py)
            
            # Dropoff reverse order (creates crossings)
            dx = 190 - (180 / n) * i
            dy = corridor_y + random.uniform(-corridor_width, corridor_width)
            coords[i + n] = (dx, dy)

    # --------------------------------------------------
    # MODE 8: TRIANGLE TRAP
    # Triangular inequality trap: direct path looks good
    # but violates constraints
    # --------------------------------------------------
    elif mode == "triangle_trap":
        for i in range(1, n + 1):
            # Pickup on left side
            px = 20 + random.uniform(-5, 5)
            py = 20 + (i - 1) * (160 / n) + random.uniform(-3, 3)
            coords[i] = (px, py)
            
            # Dropoff on right side (creates crossing)
            dx = 180 + random.uniform(-5, 5)
            dy = 180 - (i - 1) * (160 / n) + random.uniform(-3, 3)
            coords[i + n] = (dx, dy)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    dist = build_distance_matrix(coords)
    return n, k, coords, dist


def save_test_to_file(filename, n, k, dist):
    """Save test case in the required format"""
    with open(filename, 'w') as f:
        # Line 1: n and k
        f.write(f"{n} {k}\n")
        
        # Next 2n+1 lines: distance matrix
        for i in range(2 * n + 1):
            f.write(' '.join(map(str, dist[i])) + '\n')


def generate_test_suite(output_dir='test_cases'):
    """Generate a suite of test cases with varying difficulty"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Original random tests
    test_configs = [
        # (n, k, seed, description, mode)
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
    
    # Adversarial trap tests
    trap_configs = [
        # Greedy traps
        (10, 5, 100, "greedy_trap_small", "greedy_trap"),
        (20, 8, 101, "greedy_trap_medium", "greedy_trap"),
        (50, 15, 102, "greedy_trap_large", "greedy_trap"),
        
        # Cluster traps
        (15, 3, 110, "cluster_trap_small_k", "cluster_trap"),
        (30, 8, 111, "cluster_trap_medium", "cluster_trap"),
        (50, 12, 112, "cluster_trap_large", "cluster_trap"),
        
        # Capacity traps
        (10, 2, 120, "capacity_trap_k2", "capacity_trap"),
        (20, 3, 121, "capacity_trap_k3", "capacity_trap"),
        (30, 5, 122, "capacity_trap_k5", "capacity_trap"),
        
        # Symmetric traps
        (12, 6, 130, "symmetric_trap_even", "symmetric_trap"),
        (20, 8, 131, "symmetric_trap_medium", "symmetric_trap"),
        (30, 10, 132, "symmetric_trap_large", "symmetric_trap"),
        
        # Star traps (traps nearest neighbor)
        (10, 5, 140, "star_trap_small", "star_trap"),
        (20, 8, 141, "star_trap_medium", "star_trap"),
        (40, 12, 142, "star_trap_large", "star_trap"),
        
        # Corridor traps
        (15, 5, 150, "corridor_trap_small", "corridor_trap"),
        (25, 8, 151, "corridor_trap_medium", "corridor_trap"),
        (40, 10, 152, "corridor_trap_large", "corridor_trap"),
        
        # Triangle traps
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
        
        print(f"  ✓ Created: n={n_val}, k={k_val}, mode={mode}, "
              f"matrix size={(2*n_val+1)}x{(2*n_val+1)}")
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_configs)} test cases in '{output_dir}/'")
    print(f"  - {len(test_configs)} random tests")
    print(f"  - {len(trap_configs)} adversarial trap tests")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example: Generate single trap test
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
    
    # Save single test
    print("\nSaving test to 'sample_trap_test.txt'...")
    save_test_to_file('sample_trap_test.txt', n, k, dist)
    
    # Generate full test suite
    print("\n" + "="*60)
    print("Generating full test suite (random + traps)...")
    print("="*60)
    generate_test_suite()