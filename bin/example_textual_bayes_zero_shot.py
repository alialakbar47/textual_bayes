# general imports
import sys
import os
import pandas as pd
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
from bpe import BayesPE
import constants
import evaluation


# Get the data and prompt formatting for the task; we are going to be using sentiment analysis on Amazon reviews as an example
task_name = 'amazon_reviews'
n_val = 100
n_test = 200
# Load the data
df = pd.read_csv(os.path.join(path_to_package, 'data', task_name, 'test.csv'), sep='\t')  # all data
# Extract a validation and a test sets
df_val = df[:n_val]  # validation split
df_test = df[n_val:n_val+n_test]  # test split
samples_val = df_val[constants.TEXT].values  # text inputs
gt_labels_val = df_val[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
samples_test = df_test[constants.TEXT].values  # text inputs
gt_labels_test = df_test[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
# Get the prompt formatting functions for this task
sys.path.append(os.path.join(path_to_package, 'data', task_name))
import prompts  # script with prompt formatting functions

# For Textual Bayes, we start with a single initial prompt instead of multiple instructions
# The MCMC sampler will explore the prompt space starting from this initial prompt
initial_instruction = 'classify the sentiment of the Amazon review below into one of the following classes:'

# Alternative: you can still provide multiple instructions, and the first one will be used as initial prompt
instructions = [
    'classify the sentiment of the Amazon review below into one of the following classes:',
    'Categorize the sentiment of the Amazon review provided into one of the following classes:',
    'Determine the sentiment category of the given Amazon review by classifying it into one of the following classes:'
]

# Define the Textual Bayes classifier with use_textual_bayes=True
# This enables MCMC sampling instead of weight optimization
textual_bayes_classifier = BayesPE(
    model_name="google/gemma-7b-it",
    prompt_formatting=prompts,
    instructions=[initial_instruction],  # Only need one initial prompt
    use_reduced_precision=True,
    n_iterations_weights_optimiser=10,  # This maps to MCMC steps (10 * 10 = 100 steps)
    use_textual_bayes=True  # Enable Textual Bayes mode
)

# let's print out an example of the initial prompt
print("=" * 80)
print("INITIAL PROMPT EXAMPLE:")
print("=" * 80)
textual_bayes_classifier.print_prompt_example()
print("=" * 80)

# Run MCMC sampling to generate a posterior distribution of prompts
# This replaces the weight optimization step in BayesPE
print("\n" + "=" * 80)
print("STARTING MCMC SAMPLING (TEXTUAL BAYES)")
print("=" * 80)
textual_bayes_classifier.optimise_weights(samples_val, gt_labels_val)

# Save the sampled prompts (instead of weights)
textual_bayes_classifier.save_weights()

# Print some statistics about the sampled prompts
if hasattr(textual_bayes_classifier, '_textual_bayes'):
    stats = textual_bayes_classifier._textual_bayes.get_prompt_diversity_stats()
    print("\n" + "=" * 80)
    print("PROMPT DIVERSITY STATISTICS:")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 80)

# Run inference on the test set using discrete output aggregation
# This uses all sampled prompts to generate predictions with uncertainty
print("\n" + "=" * 80)
print("RUNNING INFERENCE WITH UNCERTAINTY QUANTIFICATION")
print("=" * 80)
output_probs = textual_bayes_classifier.forward(samples_test, n_forward_passes=5)

# look at some output examples and ground-truth labels
print('\nOutput probabilities (first 10 samples):')
print(output_probs[:10, :])
print('\nGround-truth labels (first 10):')
print(gt_labels_test[:10])

# evaluate output
print("\n" + "=" * 80)
print("EVALUATION METRICS:")
print("=" * 80)
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, 'f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, 'ece')
acc = evaluation.compute_metric(gt_labels_test, output_probs, 'acc')
print(f'Accuracy: {acc:.4f}')
print(f'F1-score: {f1_score:.4f}')
print(f'ECE (Expected Calibration Error): {ece:.4f}')
print("=" * 80)

# Example: Show how to print a sampled prompt
if textual_bayes_classifier.sampled_prompts and len(textual_bayes_classifier.sampled_prompts) > 0:
    print("\n" + "=" * 80)
    print("EXAMPLE SAMPLED PROMPT (first from chain):")
    print("=" * 80)
    textual_bayes_classifier.print_prompt_example(index=1)
    print("=" * 80)
