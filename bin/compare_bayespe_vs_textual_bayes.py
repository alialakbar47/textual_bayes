"""
Comparison Script: BayesPE vs Textual Bayes

This script runs both approaches on the same data and compares results.
"""

import sys
import os
import pandas as pd
import time

path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))

from bpe import BayesPE
import constants
import evaluation


def compare_methods():
    """Compare BayesPE and Textual Bayes on the same task."""
    
    # Load data
    task_name = 'amazon_reviews'
    n_val = 50  # Use smaller dataset for comparison
    n_test = 50
    
    df = pd.read_csv(os.path.join(path_to_package, 'data', task_name, 'test.csv'), sep='\t')
    df_val = df[:n_val]
    df_test = df[n_val:n_val+n_test]
    
    samples_val = df_val[constants.TEXT].values
    gt_labels_val = df_val[constants.GROUND_TRUTH_LABEL].values.astype(int)
    samples_test = df_test[constants.TEXT].values
    gt_labels_test = df_test[constants.GROUND_TRUTH_LABEL].values.astype(int)
    
    # Get prompt formatting
    sys.path.append(os.path.join(path_to_package, 'data', task_name))
    import prompts
    
    # Define prompts
    instructions_bayespe = [
        'classify the sentiment of the Amazon review below into one of the following classes:',
        'Categorize the sentiment of the Amazon review provided into one of the following classes:',
        'Determine the sentiment category of the given Amazon review by classifying it into one of the following classes:',
        'Classify the sentiment of the given Amazon review into one of the following categories:'
    ]
    
    initial_prompt_textual_bayes = instructions_bayespe[0]
    
    print("=" * 100)
    print("COMPARISON: BayesPE vs Textual Bayes")
    print("=" * 100)
    
    # ============================================================================
    # METHOD 1: Original BayesPE
    # ============================================================================
    print("\n" + "=" * 100)
    print("METHOD 1: BAYESPE (Weight Optimization)")
    print("=" * 100)
    
    bayespe = BayesPE(
        model_name="google/gemma-7b-it",
        prompt_formatting=prompts,
        instructions=instructions_bayespe,
        use_reduced_precision=True,
        n_iterations_weights_optimiser=5,
        use_textual_bayes=False  # Original BayesPE
    )
    
    print(f"\nUsing {len(instructions_bayespe)} fixed prompts")
    
    # Train
    start_time = time.time()
    bayespe.optimise_weights(samples_val, gt_labels_val)
    train_time_bayespe = time.time() - start_time
    
    print(f"\nLearned weights: {bayespe.weights}")
    print(f"Training time: {train_time_bayespe:.2f}s")
    
    # Inference
    start_time = time.time()
    output_bayespe = bayespe.forward(samples_test)
    inference_time_bayespe = time.time() - start_time
    
    print(f"Inference time: {inference_time_bayespe:.2f}s")
    
    # Evaluate
    f1_bayespe = evaluation.compute_metric(gt_labels_test, output_bayespe, 'f1')
    ece_bayespe = evaluation.compute_metric(gt_labels_test, output_bayespe, 'ece')
    acc_bayespe = evaluation.compute_metric(gt_labels_test, output_bayespe, 'acc')
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc_bayespe:.4f}")
    print(f"  F1-score: {f1_bayespe:.4f}")
    print(f"  ECE:      {ece_bayespe:.4f}")
    
    # ============================================================================
    # METHOD 2: Textual Bayes
    # ============================================================================
    print("\n" + "=" * 100)
    print("METHOD 2: TEXTUAL BAYES (MCMC Sampling)")
    print("=" * 100)
    
    textual_bayes = BayesPE(
        model_name="google/gemma-7b-it",
        prompt_formatting=prompts,
        instructions=[initial_prompt_textual_bayes],
        use_reduced_precision=True,
        n_iterations_weights_optimiser=5,  # This becomes 50 MCMC steps
        use_textual_bayes=True  # Textual Bayes mode
    )
    
    print(f"\nStarting from initial prompt:")
    print(f'"{initial_prompt_textual_bayes}"')
    
    # Train (sample prompts)
    start_time = time.time()
    textual_bayes.optimise_weights(samples_val, gt_labels_val)
    train_time_tb = time.time() - start_time
    
    print(f"\nTraining time: {train_time_tb:.2f}s")
    
    # Show prompt diversity
    if hasattr(textual_bayes, '_textual_bayes'):
        stats = textual_bayes._textual_bayes.get_prompt_diversity_stats()
        print(f"Sampled {stats['total_prompts']} prompts")
        print(f"Unique prompts: {stats['unique_prompts']}")
        print(f"Diversity ratio: {stats['diversity_ratio']:.3f}")
    
    # Inference
    start_time = time.time()
    output_tb = textual_bayes.forward(samples_test)
    inference_time_tb = time.time() - start_time
    
    print(f"Inference time: {inference_time_tb:.2f}s")
    
    # Evaluate
    f1_tb = evaluation.compute_metric(gt_labels_test, output_tb, 'f1')
    ece_tb = evaluation.compute_metric(gt_labels_test, output_tb, 'ece')
    acc_tb = evaluation.compute_metric(gt_labels_test, output_tb, 'acc')
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc_tb:.4f}")
    print(f"  F1-score: {f1_tb:.4f}")
    print(f"  ECE:      {ece_tb:.4f}")
    
    # ============================================================================
    # COMPARISON SUMMARY
    # ============================================================================
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    
    print("\n{:<30s} {:>15s} {:>15s} {:>15s}".format(
        "Metric", "BayesPE", "Textual Bayes", "Difference"
    ))
    print("-" * 100)
    
    metrics = [
        ("Accuracy", acc_bayespe, acc_tb),
        ("F1-Score", f1_bayespe, f1_tb),
        ("ECE", ece_bayespe, ece_tb),
        ("Training Time (s)", train_time_bayespe, train_time_tb),
        ("Inference Time (s)", inference_time_bayespe, inference_time_tb)
    ]
    
    for metric_name, val_bayespe, val_tb in metrics:
        diff = val_tb - val_bayespe
        if metric_name in ["Accuracy", "F1-Score"]:
            # Higher is better
            symbol = "↑" if diff > 0 else "↓"
        elif metric_name == "ECE":
            # Lower is better
            symbol = "↓" if diff < 0 else "↑"
        else:
            symbol = ""
        
        print("{:<30s} {:>15.4f} {:>15.4f} {:>14.4f} {}".format(
            metric_name, val_bayespe, val_tb, diff, symbol
        ))
    
    print("\n" + "=" * 100)
    print("KEY DIFFERENCES")
    print("=" * 100)
    print("""
BayesPE:
  ✓ Optimizes weights over fixed prompts
  ✓ Faster training (no prompt generation)
  ✓ Deterministic results
  ✓ Weighted probability averaging
  ✓ Requires manually designed prompt variations

Textual Bayes:
  ✓ Explores prompt space with MCMC
  ✓ Discovers new effective prompts
  ✓ Discrete output aggregation
  ✓ Natural uncertainty quantification
  ✓ Requires only one initial prompt
  ✗ Slower due to prompt generation
  
Both methods provide calibrated uncertainty estimates and can be used
for the same downstream tasks.
    """)
    
    print("=" * 100)


if __name__ == "__main__":
    compare_methods()
