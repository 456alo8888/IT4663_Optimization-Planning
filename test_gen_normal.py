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

def save_test_to_file(filename, n, k, dist):
    with open(filename, 'w') as f:
        f.write(f"{n} {k}\n")
        for i in range(2 * n + 1):
            f.write(' '.join(map(str, dist[i])) + '\n')

def generate_test_suite(output_dir='test_cases'):
    os.makedirs(output_dir, exist_ok=True)
    test_configs = [
        (3, 2, 42, "tiny"),
        (5, 2, 43, "small"),
        (10, 3, 44, "medium_low_capacity"),
        (10, 8, 45, "medium_high_capacity"),
        (20, 5, 46, "medium"),
        (50, 10, 47, "large"),
        (100, 20, 48, "very_large"),
        (200, 30, 49, "huge"),
        (500, 40, 50, "massive"),
        (1000, 50, 51, "extreme"),
    ]
    for i, (n, k, seed, desc) in enumerate(test_configs, 1):
        filename = os.path.join(output_dir, f"test{i:02d}_{desc}_n{n}_k{k}.txt")
        print(f"Generating {filename}...")
        n_val, k_val, coords, dist = generate_cbus_test(n=n, k=k, seed=seed)
        save_test_to_file(filename, n_val, k_val, dist)
        print(f"  ✓ Created: n={n_val}, k={k_val}, matrix size={(2*n_val+1)}x{(2*n_val+1)}")
    print(f"\nGenerated {len(test_configs)} test cases in '{output_dir}/'")

if __name__ == "__main__":
    n, k, coords, dist = generate_cbus_test(n=3, k=2, seed=42)
    print(f"n={n}, k={k}")
    print("\nCoordinates:")
    print("  0 (depot):", coords[0])
    for i in range(1, n + 1):
        print(f"  {i} (pickup): {coords[i]}")
    for i in range(n + 1, 2 * n + 1):
        print(f"  {i} (dropoff): {coords[i]}")
    print("\nDistance Matrix (first 5 rows):")
    for i in range(min(5, 2*n + 1)):
        print(f"  Row {i}:", ' '.join(f"{d:4d}" for d in dist[i][:min(10, 2*n+1)]))
    print("\nSaving test to 'sample_test.txt'...")
    save_test_to_file('sample_test.txt', n, k, dist)
    print("\n" + "="*60)
    print("Generating full test suite...")
    print("="*60)
    generate_test_suite()