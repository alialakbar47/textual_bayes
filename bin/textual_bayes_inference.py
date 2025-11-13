"""
Textual Bayes Inference Script

This script demonstrates how to use pre-sampled prompts for inference
with uncertainty quantification through discrete output aggregation.
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))

from bpe import BayesPE
import constants
import evaluation


def load_and_predict():
    """
    Load pre-trained Textual Bayes model and generate predictions with uncertainty.
    """
    # Get the data
    task_name = 'amazon_reviews'
    n_test = 200
    
    df = pd.read_csv(os.path.join(path_to_package, 'data', task_name, 'test.csv'), sep='\t')
    df_test = df[:n_test]
    samples_test = df_test[constants.TEXT].values
    gt_labels_test = df_test[constants.GROUND_TRUTH_LABEL].values.astype(int)
    
    # Get prompt formatting
    sys.path.append(os.path.join(path_to_package, 'data', task_name))
    import prompts
    
    # Initialize Textual Bayes in inference mode
    print("=" * 80)
    print("LOADING TEXTUAL BAYES MODEL")
    print("=" * 80)
    
    # We don't need to specify the full configuration since we're just loading
    classifier = BayesPE(
        model_name="google/gemma-7b-it",
        prompt_formatting=prompts,
        instructions=['placeholder'],  # Will be replaced by loaded prompts
        use_reduced_precision=True,
        use_textual_bayes=True
    )
    
    # Load the sampled prompts
    try:
        classifier.load_weights()  # Loads from saved_prompts/prompt_chain.json
        print(f"✓ Loaded {len(classifier.sampled_prompts)} sampled prompts")
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        print("Make sure you've run the training script first!")
        return
    
    # Print some example prompts
    print("\n" + "=" * 80)
    print("EXAMPLE SAMPLED PROMPTS:")
    print("=" * 80)
    if classifier.sampled_prompts:
        for i in range(min(3, len(classifier.sampled_prompts))):
            print(f"\nPrompt {i+1}:")
            print("-" * 80)
            print(classifier.sampled_prompts[i])
    
    # Generate predictions with uncertainty
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS WITH UNCERTAINTY")
    print("=" * 80)
    print(f"Using {len(classifier.sampled_prompts)} prompts for inference...")
    
    output_probs = classifier.forward(samples_test)
    
    # Analyze uncertainty for some examples
    print("\n" + "=" * 80)
    print("UNCERTAINTY ANALYSIS (First 5 samples):")
    print("=" * 80)
    
    class_names = prompts.CLASSES
    
    for i in range(min(5, len(samples_test))):
        print(f"\nSample {i+1}:")
        print(f"Text: {samples_test[i][:100]}...")
        print(f"True Label: {class_names[gt_labels_test[i]]}")
        print(f"Predicted: {class_names[np.argmax(output_probs[i])]}")
        print(f"Confidence Distribution:")
        for j, class_name in enumerate(class_names):
            bar = '█' * int(output_probs[i, j] * 50)
            print(f"  {class_name:15s}: {output_probs[i, j]:.3f} {bar}")
        
        # Calculate entropy as uncertainty measure
        probs = output_probs[i]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = -np.log(1.0 / len(probs))
        normalized_entropy = entropy / max_entropy
        print(f"Uncertainty (normalized entropy): {normalized_entropy:.3f}")
    
    # Compute overall metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS:")
    print("=" * 80)
    
    f1 = evaluation.compute_metric(gt_labels_test, output_probs, 'f1')
    ece = evaluation.compute_metric(gt_labels_test, output_probs, 'ece')
    acc = evaluation.compute_metric(gt_labels_test, output_probs, 'acc')
    nll = evaluation.compute_metric(gt_labels_test, output_probs, 'nll')
    
    print(f"Accuracy:                      {acc:.4f}")
    print(f"F1-score (macro):              {f1:.4f}")
    print(f"Expected Calibration Error:    {ece:.4f}")
    print(f"Negative Log-Likelihood:       {nll:.4f}")
    
    # Analyze calibration
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS:")
    print("=" * 80)
    
    predicted_labels = np.argmax(output_probs, axis=1)
    max_confidences = np.max(output_probs, axis=1)
    correct = predicted_labels == gt_labels_test
    
    # Bin by confidence
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    for i in range(len(bins) - 1):
        bin_mask = (max_confidences >= bins[i]) & (max_confidences < bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(correct[bin_mask])
            bin_conf = np.mean(max_confidences[bin_mask])
            bin_count = np.sum(bin_mask)
            print(f"Confidence [{bins[i]:.2f}, {bins[i+1]:.2f}): "
                  f"Accuracy = {bin_acc:.3f}, "
                  f"Avg Conf = {bin_conf:.3f}, "
                  f"Count = {bin_count}")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    load_and_predict()
