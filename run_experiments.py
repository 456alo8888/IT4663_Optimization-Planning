"""
Script to run GA and SA experiments with different configurations
"""
import subprocess
import os
import glob

def run_experiment(input_file, method, output_suffix, **params):
    """Run a single experiment with given parameters"""
    test_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"output/{test_name}_{output_suffix}.txt"
    
    cmd = ["python", "run_GA.py", "-i", input_file, "-o", output_file, "-m", method]
    
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_ga_experiments(input_file):
    """Run GA with different configurations"""
    print(f"\n{'='*70}")
    print(f"Running GA experiments for {os.path.basename(input_file)}")
    print('='*70)
    
    configs = [
        # Small population, more generations
        {
            'suffix': 'ga_small_pop',
            'params': {'pop_size': 50, 'generations': 400, 'mutation_rate': 0.2}
        },
        # Medium population (default)
        {
            'suffix': 'ga_medium_pop',
            'params': {'pop_size': 100, 'generations': 200, 'mutation_rate': 0.2}
        },
        # Large population, fewer generations
        {
            'suffix': 'ga_large_pop',
            'params': {'pop_size': 200, 'generations': 100, 'mutation_rate': 0.2}
        },
        # Higher mutation rate
        {
            'suffix': 'ga_high_mutation',
            'params': {'pop_size': 100, 'generations': 200, 'mutation_rate': 0.4}
        },
        # Lower mutation rate
        {
            'suffix': 'ga_low_mutation',
            'params': {'pop_size': 100, 'generations': 200, 'mutation_rate': 0.1}
        },
    ]
    
    for config in configs:
        run_experiment(input_file, 'ga', config['suffix'], **config['params'])

def run_sa_experiments(input_file):
    """Run SA with different configurations"""
    print(f"\n{'='*70}")
    print(f"Running SA experiments for {os.path.basename(input_file)}")
    print('='*70)
    
    configs = [
        # High temperature, slow cooling
        {
            'suffix': 'sa_high_temp_slow',
            'params': {'sa_T0': 2000, 'sa_alpha': 0.999, 'sa_max_iter': 10000, 'sa_coef': 100}
        },
        # Medium temperature (default)
        {
            'suffix': 'sa_medium',
            'params': {'sa_T0': 1000, 'sa_alpha': 0.995, 'sa_max_iter': 5000, 'sa_coef': 100}
        },
        # Low temperature, fast cooling
        {
            'suffix': 'sa_low_temp_fast',
            'params': {'sa_T0': 500, 'sa_alpha': 0.99, 'sa_max_iter': 3000, 'sa_coef': 100}
        },
        # Higher violation penalty
        {
            'suffix': 'sa_high_penalty',
            'params': {'sa_T0': 1000, 'sa_alpha': 0.995, 'sa_max_iter': 5000, 'sa_coef': 500}
        },
        # Lower violation penalty
        {
            'suffix': 'sa_low_penalty',
            'params': {'sa_T0': 1000, 'sa_alpha': 0.995, 'sa_max_iter': 5000, 'sa_coef': 50}
        },
    ]
    
    for config in configs:
        run_experiment(input_file, 'sa', config['suffix'], **config['params'])

def compare_methods(input_file):
    """Compare GA vs SA with default parameters"""
    print(f"\n{'='*70}")
    print(f"Comparing GA vs SA for {os.path.basename(input_file)}")
    print('='*70)
    
    # Run GA
    run_experiment(input_file, 'ga', 'ga_default', 
                  pop_size=100, generations=200, mutation_rate=0.2)
    
    # Run SA
    run_experiment(input_file, 'sa', 'sa_default',
                  sa_T0=1000, sa_alpha=0.995, sa_max_iter=5000, sa_coef=100)

def main():
    """Main function to run experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experiments with different configs')
    parser.add_argument('-i', '--input', type=str, help='Specific input file')
    parser.add_argument('-d', '--directory', type=str, default='test_cases', 
                       help='Directory containing test files')
    parser.add_argument('-e', '--experiment', type=str, 
                       choices=['ga', 'sa', 'compare', 'all'], 
                       default='compare',
                       help='Type of experiment to run')
    parser.add_argument('-p', '--pattern', type=str, default='*.txt',
                       help='File pattern to match (e.g., test*.txt)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    if args.input:
        files = [args.input]
    else:
        pattern = os.path.join(args.directory, args.pattern)
        files = glob.glob(pattern)
        files.sort()
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} test file(s)")
    
    for input_file in files:
        if args.experiment == 'ga':
            run_ga_experiments(input_file)
        elif args.experiment == 'sa':
            run_sa_experiments(input_file)
        elif args.experiment == 'compare':
            compare_methods(input_file)
        elif args.experiment == 'all':
            run_ga_experiments(input_file)
            run_sa_experiments(input_file)
            compare_methods(input_file)

if __name__ == '__main__':
    main()
