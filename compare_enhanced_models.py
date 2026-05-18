"""
Compare Enhanced Model Results
Generates comparison tables for all enhanced classical ML models
"""

import pandas as pd
import os

print("="*90)
print("ENHANCED MODELS COMPARISON")
print("="*90)

# Models to compare
enhanced_models = [
    'LogisticRegression_Enhanced',
    'RandomForest_Enhanced',
    'XGBoost_Enhanced',
    'MLP_Enhanced'
]

# Load all enhanced results
results_list = []
hyperparams_list = []

for model in enhanced_models:
    result_file = f'results/{model}_results.csv'
    hyperparam_file = f'results/{model}_hyperparameters.csv'
    
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        results_list.append(df)
        print(f"✅ Loaded: {result_file}")
        
        if os.path.exists(hyperparam_file):
            hp_df = pd.read_csv(hyperparam_file)
            hp_df.insert(0, 'Model', model.replace('_Enhanced', ''))
            hyperparams_list.append(hp_df)
    else:
        print(f"❌ Not found: {result_file}")

if not results_list:
    print("\n⚠️  No enhanced model results found. Please run the enhanced models first.")
    exit()

# Combine all results
all_results = pd.concat(results_list, ignore_index=True)

# Separate regression and classification
regression_results = all_results[all_results['Model'].str.contains('Regression', na=False) & 
                                 ~all_results['Model'].str.contains('Classification', na=False)]
classification_results = all_results[all_results['Model'].str.contains('Classification', na=False)]

# Clean up model names for display
classification_results['Model'] = classification_results['Model'].str.replace('_Classification_Enhanced', '')
regression_results['Model'] = regression_results['Model'].str.replace('_Regression', '')

# ================= REGRESSION COMPARISON =================
print("\n" + "="*90)
print("REGRESSION RESULTS - ENHANCED MODELS COMPARISON")
print("="*90)

print("\n10-Fold Cross-Validation Results:")
print("-" * 90)

cv_reg_table = regression_results[['Model', 'CV_R/Acc', 'CV_RMSE']].copy()
cv_reg_table.columns = ['Model', 'R', 'RMSE']
cv_reg_table = cv_reg_table.sort_values('R', ascending=False).reset_index(drop=True)
cv_reg_table.index = cv_reg_table.index + 1

print(cv_reg_table.to_string(index=True, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))

print("\n\nTest Set Results:")
print("-" * 90)

test_reg_table = regression_results[['Model', 'Test_R/Acc', 'Test_RMSE']].copy()
test_reg_table.columns = ['Model', 'R', 'RMSE']
test_reg_table = test_reg_table.sort_values('R', ascending=False).reset_index(drop=True)
test_reg_table.index = test_reg_table.index + 1

print(test_reg_table.to_string(index=True, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))

best_model_reg = test_reg_table.iloc[0]
print(f"\n🏆 Best Regression Model: {best_model_reg['Model']} (R = {best_model_reg['R']:.4f}, RMSE = {best_model_reg['RMSE']:.4f})")

# ================= CLASSIFICATION COMPARISON =================
print("\n\n" + "="*120)
print("CLASSIFICATION RESULTS - ENHANCED MODELS COMPARISON")
print("="*120)

print("\n10-Fold Cross-Validation Results:")
print("-" * 120)

cv_class_table = classification_results[['Model', 'CV_R/Acc', 'CV_Sens', 'CV_Spec', 'CV_F1', 'CV_AUC', 'Optimal_Threshold']].copy()
cv_class_table.columns = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC', 'Threshold']
cv_class_table = cv_class_table.sort_values('AUC', ascending=False).reset_index(drop=True)
cv_class_table.index = cv_class_table.index + 1

print(cv_class_table.to_string(index=True, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))

print("\n\nTest Set Results:")
print("-" * 120)

test_class_table = classification_results[['Model', 'Test_R/Acc', 'Test_Sens', 'Test_Spec', 'Test_F1', 'Test_AUC', 'Optimal_Threshold']].copy()
test_class_table.columns = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC', 'Threshold']
test_class_table = test_class_table.sort_values('AUC', ascending=False).reset_index(drop=True)
test_class_table.index = test_class_table.index + 1

print(test_class_table.to_string(index=True, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))

best_model_class = test_class_table.iloc[0]
print(f"\n🏆 Best Classification Model: {best_model_class['Model']} "
      f"(Accuracy = {best_model_class['Accuracy']:.4f}, AUC = {best_model_class['AUC']:.4f}, "
      f"Threshold = {best_model_class['Threshold']})")

# ================= SAVE COMPARISON RESULTS =================
print("\n\n" + "="*90)
print("SAVING COMPARISON RESULTS")
print("="*90)

# Save to CSV
os.makedirs('results', exist_ok=True)

regression_comparison = cv_reg_table.copy()
regression_comparison.columns = ['Model', 'CV_R', 'CV_RMSE']
test_reg_table_copy = test_reg_table.copy()
test_reg_table_copy.columns = ['Model', 'Test_R', 'Test_RMSE']
regression_comparison = pd.merge(regression_comparison, test_reg_table_copy, on='Model')
regression_comparison.to_csv('results/Enhanced_Regression_Comparison.csv', index=False)
print("✅ Saved: results/Enhanced_Regression_Comparison.csv")

classification_comparison = cv_class_table.copy()
classification_comparison.columns = ['Model', 'CV_Acc', 'CV_Sens', 'CV_Spec', 'CV_F1', 'CV_AUC', 'CV_Threshold']
test_class_table_copy = test_class_table.copy()
test_class_table_copy.columns = ['Model', 'Test_Acc', 'Test_Sens', 'Test_Spec', 'Test_F1', 'Test_AUC', 'Test_Threshold']
classification_comparison = pd.merge(classification_comparison, test_class_table_copy, on='Model')
classification_comparison.to_csv('results/Enhanced_Classification_Comparison.csv', index=False)
print("✅ Saved: results/Enhanced_Classification_Comparison.csv")

# Save hyperparameters comparison
if hyperparams_list:
    all_hyperparams = pd.concat(hyperparams_list, ignore_index=True)
    all_hyperparams.to_csv('results/Enhanced_Models_Hyperparameters.csv', index=False)
    print("✅ Saved: results/Enhanced_Models_Hyperparameters.csv")

# Save detailed text report
output_file = 'results/Enhanced_Models_Comparison_Report.txt'
with open(output_file, 'w') as f:
    f.write("="*90 + "\n")
    f.write("ENHANCED MODELS COMPARISON REPORT\n")
    f.write("="*90 + "\n\n")
    
    f.write("REGRESSION RESULTS\n")
    f.write("="*90 + "\n\n")
    f.write("10-Fold Cross-Validation:\n")
    f.write("-" * 90 + "\n")
    f.write(cv_reg_table.to_string(float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    f.write("\n\nTest Set:\n")
    f.write("-" * 90 + "\n")
    f.write(test_reg_table.to_string(float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    f.write(f"\n\nBest: {best_model_reg['Model']} (R = {best_model_reg['R']:.4f}, RMSE = {best_model_reg['RMSE']:.4f})\n")
    
    f.write("\n\n" + "="*120 + "\n")
    f.write("CLASSIFICATION RESULTS (WITH OPTIMIZED THRESHOLDS)\n")
    f.write("="*120 + "\n\n")
    f.write("10-Fold Cross-Validation:\n")
    f.write("-" * 120 + "\n")
    f.write(cv_class_table.to_string(float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    f.write("\n\nTest Set:\n")
    f.write("-" * 120 + "\n")
    f.write(test_class_table.to_string(float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    f.write(f"\n\nBest: {best_model_class['Model']} "
            f"(Accuracy = {best_model_class['Accuracy']:.4f}, AUC = {best_model_class['AUC']:.4f}, "
            f"Threshold = {best_model_class['Threshold']})\n")
    
    f.write("\n\n" + "="*90 + "\n")
    f.write("KEY IMPROVEMENTS\n")
    f.write("="*90 + "\n")
    f.write("✅ Automatic threshold optimization using ROC curve (Youden Index)\n")
    f.write("✅ StratifiedKFold for classification cross-validation\n")
    f.write("✅ Hyperparameter tuning with RandomizedSearchCV\n")
    f.write("✅ Balanced class weights for handling imbalanced data\n")
    f.write("✅ Comprehensive metrics reporting\n")

print(f"✅ Saved: {output_file}")

print("\n" + "="*90)
print("COMPARISON COMPLETE!")
print("="*90)
