# Textual Bayes Implementation Guide

## Overview

This codebase now supports **Textual Bayes**, a Bayesian inference framework for LLMs that uses MCMC sampling to explore the space of prompts. This is a fundamental shift from the original BayesPE approach:

- **BayesPE**: Optimizes weights over a _fixed set_ of prompts
- **Textual Bayes**: Uses MCMC to _explore the prompt space_, generating a posterior distribution of effective prompts

## Key Components

### 1. MHLP Sampler (`src/mhlp_sampler.py`)

The core MCMC algorithm: **Metropolis-Hastings through LLM Proposals (MHLP)**

**Key Functions:**

- `run_mhlp()`: Main MCMC loop that generates a chain of prompts
- `_propose_new_prompt()`: Uses an LLM to propose improved prompts
- `_log_p_D_given_theta()`: Calculates likelihood of data given a prompt
- `_log_p_theta()`: Evaluates prior probability based on textual constraints

**Algorithm:**

```python
for t in range(num_steps):
    # 1. Propose new prompt using LLM
    proposed_prompt = propose_new_prompt(current_prompt, train_data)

    # 2. Calculate acceptance probability
    log_acceptance_ratio = log_g(proposed) - log_g(current)
    acceptance_prob = min(1.0, exp(log_acceptance_ratio))

    # 3. Accept or reject
    if random() < acceptance_prob:
        current_prompt = proposed_prompt
```

### 2. Textual Bayes Class (`src/textual_bayes.py`)

The main interface for Textual Bayes, providing:

- Prompt sampling via MCMC
- Prediction with uncertainty through discrete output aggregation
- Backward compatibility with BayesPE API

**Key Methods:**

- `sample_prompts()`: Run MCMC to generate posterior prompts
- `predict_with_uncertainty()`: Generate predictions using sampled prompts
- `forward()`: Compatible interface with BayesPE

### 3. Updated BayesPE (`src/bpe.py`)

The original BayesPE class now supports both modes:

- **Original BayesPE mode** (default): Weight optimization
- **Textual Bayes mode**: MCMC sampling (set `use_textual_bayes=True`)

## Usage

### Basic Textual Bayes Example

```python
from bpe import BayesPE
import prompts  # Your task's prompt formatting

# Define initial prompt (only need one, not multiple)
initial_prompt = 'classify the sentiment of the review into one of the following classes:'

# Create Textual Bayes classifier
classifier = BayesPE(
    model_name="google/gemma-7b-it",
    prompt_formatting=prompts,
    instructions=[initial_prompt],
    use_textual_bayes=True,  # Enable Textual Bayes mode
    n_iterations_weights_optimiser=10  # Maps to 100 MCMC steps (10 * 10)
)

# Run MCMC sampling (replaces weight optimization)
classifier.optimise_weights(train_texts, train_labels)

# Save sampled prompts
classifier.save_weights()  # Saves to 'saved_prompts/prompt_chain.json'

# Generate predictions with uncertainty
predictions = classifier.forward(test_texts, n_forward_passes=5)
```

### Advanced: Direct Textual Bayes API

```python
from textual_bayes import TextualBayes

# More control over MCMC parameters
tb = TextualBayes(
    model_name="google/gemma-7b-it",
    prompt_formatting=prompts,
    initial_prompt=initial_prompt,
    num_mcmc_steps=200,
    burn_in=40,
    thinning=10
)

# Sample prompts with prior constraints
sampled_prompts = tb.sample_prompts(
    input_texts=train_texts,
    gt_labels=train_labels,
    prior_description="The prompt should be concise and use simple language"
)

# Get predictions with uncertainty
predicted_labels, confidence_dists = tb.predict_with_uncertainty(test_texts)

# Analyze prompt diversity
stats = tb.get_prompt_diversity_stats()
print(f"Unique prompts: {stats['unique_prompts']}")
print(f"Diversity ratio: {stats['diversity_ratio']:.2f}")
```

## Key Differences from BayesPE

| Aspect          | BayesPE                        | Textual Bayes               |
| --------------- | ------------------------------ | --------------------------- |
| **Input**       | Multiple fixed prompts         | Single initial prompt       |
| **Training**    | Weight optimization (L-BFGS)   | MCMC sampling (MHLP)        |
| **Output**      | Weighted prompt ensemble       | Sampled prompt chain        |
| **Inference**   | Weighted probability averaging | Discrete output aggregation |
| **Uncertainty** | Weighted mixture               | Frequency distribution      |

## Inference & Uncertainty

### BayesPE Approach

```python
# Weighted average of probabilities
p(y|x) = Σ w_i * p(y|x, prompt_i)
```

### Textual Bayes Approach

```python
# Discrete output aggregation
for each sampled_prompt:
    outputs.append(generate_discrete_output(prompt, x))

# Confidence from frequency distribution
confidence[y] = count(outputs == y) / total_outputs
```

**Benefits:**

- Natural uncertainty quantification
- Works for generative tasks (not just classification)
- Captures epistemic uncertainty over prompt formulations

## File Structure

```
src/
├── bpe.py                    # Main class (supports both modes)
├── textual_bayes.py          # Pure Textual Bayes implementation
├── mhlp_sampler.py           # MCMC sampler
├── llm_classifier.py         # LLM interface (unchanged)
├── llm_model.py              # Model wrapper (unchanged)
└── evaluation.py             # Metrics (unchanged)

bin/
├── example_textual_bayes_zero_shot.py    # New Textual Bayes example
├── example_3_zero_shot_bayespe.py        # Original BayesPE example
└── ...

saved_prompts/               # Stores sampled prompt chains
└── prompt_chain.json
```

## Configuration Parameters

### MCMC Parameters

- `num_mcmc_steps`: Total MCMC iterations (default: 100)
- `burn_in`: Initial samples to discard (default: 20)
- `thinning`: Keep every nth sample (default: 5)
- `batch_size`: Training examples per likelihood evaluation (default: 10)

**Example:**

```python
# 200 steps, discard first 40, keep every 10th
# Final chain: (200 - 40) / 10 = 16 prompts
tb = TextualBayes(
    num_mcmc_steps=200,
    burn_in=40,
    thinning=10
)
```

### Prior Constraints

You can guide the prompt search with textual priors:

```python
prior_desc = """
The prompt should:
1. Be concise (under 50 words)
2. Use simple, clear language
3. Include explicit instructions for the output format
"""

sampled_prompts = tb.sample_prompts(
    train_data,
    train_labels,
    prior_description=prior_desc
)
```

## Migration from BayesPE

### Minimal Changes

```python
# Just add one parameter!
classifier = BayesPE(
    model_name="...",
    prompt_formatting=prompts,
    instructions=[initial_prompt],  # Only need 1 prompt now
    use_textual_bayes=True  # Enable Textual Bayes
)
```

### Global Toggle

```python
# In src/bpe.py, set:
USE_TEXTUAL_BAYES = True

# Now all BayesPE instances use Textual Bayes by default
```

## Running Examples

### Textual Bayes Zero-Shot

```bash
python bin/example_textual_bayes_zero_shot.py
```

### Original BayesPE (still works)

```bash
python bin/example_3_zero_shot_bayespe.py
```

## Evaluation Metrics

Both modes support the same metrics:

- **F1 Score**: Macro-averaged F1
- **Accuracy**: Classification accuracy
- **ECE**: Expected Calibration Error
- **ROC-AUC**: Area under ROC curve

```python
import evaluation

# Works for both BayesPE and Textual Bayes outputs
f1 = evaluation.compute_metric(gt_labels, predictions, 'f1')
ece = evaluation.compute_metric(gt_labels, predictions, 'ece')
acc = evaluation.compute_metric(gt_labels, predictions, 'acc')
```

## Troubleshooting

### Low Acceptance Rate

If acceptance rate is very low (<10%):

- Reduce batch_size for faster likelihood computation
- Adjust prior_description to be less restrictive
- Increase temperature in proposal generation

### Prompts Not Changing

If all prompts in chain are identical:

- Check that proposal function is working
- Verify LLM is generating varied outputs
- Increase num_mcmc_steps
- Try different initial_prompt

### Out of Memory

For large models:

- Set `use_reduced_precision=True`
- Reduce `batch_size` in MHLPSampler
- Use smaller model for proposal generation

## Advanced Features

### Custom Proposal Function

```python
# Override proposal in MHLPSampler
class CustomMHLP(MHLPSampler):
    def _propose_new_prompt(self, current_prompt, train_data):
        # Your custom logic here
        return new_prompt
```

### Semantic Clustering for Generative Tasks

For open-ended generation, group semantically similar outputs:

```python
# In textual_bayes.py, modify predict_with_uncertainty()
from sentence_transformers import SentenceTransformer

def semantic_clustering(outputs):
    embeddings = model.encode(outputs)
    # Cluster and aggregate
    return clustered_outputs
```

## References

- **Textual Bayes Paper**: [Link to paper PDF]
- **BayesPE Paper**: Original repository
- **MCMC Methods**: Metropolis-Hastings algorithm
- **Prompt Optimization**: Automatic Prompt Engineer (APE)

## Support

For issues or questions:

1. Check existing examples in `bin/`
2. Review this guide
3. Examine the code comments in `src/`
4. Open an issue on the repository

---

**Last Updated**: Implementation complete with MHLP sampler, Textual Bayes class, and example scripts.
