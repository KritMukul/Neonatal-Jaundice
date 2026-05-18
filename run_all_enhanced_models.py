"""
Run All Enhanced Models - Batch Script
Runs all enhanced classical ML models sequentially and generates comparison report
"""

import subprocess
import sys
import time

print("="*80)
print("RUNNING ALL ENHANCED MODELS")
print("="*80)

models = [
    ('LogisticRegression_Enhanced.py', 'Logistic Regression'),
    ('RandomForest_Enhanced.py', 'Random Forest'),
    ('XGBoost_Enhanced.py', 'XGBoost'),
    ('MLP_Enhanced.py', 'Multi-Layer Perceptron')
]

results = []
total_start_time = time.time()

for script, name in models:
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Script: {script}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {name} completed successfully in {elapsed_time:.1f} seconds")
            results.append((name, 'SUCCESS', elapsed_time))
        else:
            print(f"\n❌ {name} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            results.append((name, 'FAILED', elapsed_time))
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  {name} timed out after {elapsed_time:.1f} seconds")
        results.append((name, 'TIMEOUT', elapsed_time))
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {name} encountered an error: {str(e)}")
        results.append((name, 'ERROR', elapsed_time))

total_elapsed_time = time.time() - total_start_time

# Print summary
print(f"\n{'='*80}")
print("BATCH EXECUTION SUMMARY")
print(f"{'='*80}\n")

for name, status, elapsed in results:
    status_icon = '✅' if status == 'SUCCESS' else '❌'
    print(f"{status_icon} {name:30s} | {status:10s} | {elapsed:.1f}s")

print(f"\nTotal execution time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.1f} minutes)")

# Run comparison if all models succeeded
successful_models = sum(1 for _, status, _ in results if status == 'SUCCESS')
if successful_models == len(models):
    print(f"\n{'='*80}")
    print("All models completed successfully! Running comparison...")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'compare_enhanced_models.py'],
            capture_output=False,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("\n✅ Comparison report generated successfully!")
        else:
            print("\n❌ Comparison report generation failed")
    except Exception as e:
        print(f"\n❌ Error generating comparison report: {str(e)}")
else:
    print(f"\n⚠️  Only {successful_models}/{len(models)} models completed successfully.")
    print("Fix the errors and re-run failed models before generating comparison report.")

print(f"\n{'='*80}")
print("BATCH EXECUTION COMPLETE!")
print(f"{'='*80}")
