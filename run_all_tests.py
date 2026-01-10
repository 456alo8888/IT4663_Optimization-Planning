"""
Script to run all test cases and log results
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ga import solve_from_file


def run_all_tests(test_dir, output_dir):
    """Run all test files in the test directory"""
    # Get all test files
    test_files = []
    for file in os.listdir(test_dir):
        if file.endswith('.txt'):
            test_files.append(os.path.join(test_dir, file))
    
    test_files.sort()
    
    print(f"Found {len(test_files)} test files")
    print("="*60)
    
    results = []
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] Processing: {os.path.basename(test_file)}")
        print("-"*60)
        
        try:
            route, cost = solve_from_file(test_file, output_dir)
            results.append({
                'file': os.path.basename(test_file),
                'status': 'SUCCESS',
                'cost': cost
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'file': os.path.basename(test_file),
                'status': 'FAILED',
                'error': str(e)
            })
        
        print("="*60)
    
    # Print summary
    print("\n\nSUMMARY")
    print("="*60)
    for result in results:
        if result['status'] == 'SUCCESS':
            print(f"✓ {result['file']:20s} - Cost: {result['cost']}")
        else:
            print(f"✗ {result['file']:20s} - {result['error']}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nTotal: {len(results)} | Success: {success_count} | Failed: {len(results) - success_count}")


if __name__ == "__main__":
    # Default paths
    test_dir = r"D:\HUST\20251\IT4663_TULKH\project\test"
    output_dir = r"D:\HUST\20251\IT4663_TULKH\project\output"
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    run_all_tests(test_dir, output_dir)
