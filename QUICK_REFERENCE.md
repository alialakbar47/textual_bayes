# Textual Bayes Quick Reference

## 🚀 Quick Start (3 Steps)

### 1. Enable Textual Bayes

```python
from bpe import BayesPE

classifier = BayesPE(
    model_name="google/gemma-7b-it",
    prompt_formatting=prompts,
    instructions=["Classify sentiment:"],  # Only 1 initial prompt needed
    use_textual_bayes=True  # ← Add this line
)
```

### 2. Train (MCMC Sampling)

```python
classifier.optimise_weights(train_texts, train_labels)
classifier.save_weights()  # Saves to 'saved_prompts/prompt_chain.json'
```

### 3. Predict

```python
predictions = classifier.forward(test_texts)
```

---

## 📁 File Quick Reference

| File                                      | Purpose                  | Key Functions                                    |
| ----------------------------------------- | ------------------------ | ------------------------------------------------ |
| `src/mhlp_sampler.py`                     | MCMC sampler             | `run_mhlp()`, `_propose_new_prompt()`            |
| `src/textual_bayes.py`                    | Main Textual Bayes class | `sample_prompts()`, `predict_with_uncertainty()` |
| `src/bpe.py`                              | Dual-mode BayesPE        | `optimise_weights()`, `forward()`                |
| `bin/example_textual_bayes_zero_shot.py`  | Full example             | -                                                |
| `bin/textual_bayes_inference.py`          | Inference only           | -                                                |
| `bin/compare_bayespe_vs_textual_bayes.py` | Comparison               | -                                                |
| `bin/validate_textual_bayes.py`           | Test suite               | -                                                |

---

## ⚙️ Key Parameters

### MCMC Configuration

```python
TextualBayes(
    num_mcmc_steps=100,   # Total MCMC iterations
    burn_in=20,           # Initial samples to discard
    thinning=5,           # Keep every 5th sample
    batch_size=10         # Examples per likelihood eval
)
# → Effective samples: (100-20)/5 = 16 prompts
```

### Quick Sizing Guide

- **Quick test**: `num_mcmc_steps=20, burn_in=5, thinning=2`
- **Standard**: `num_mcmc_steps=100, burn_in=20, thinning=5`
- **High quality**: `num_mcmc_steps=200, burn_in=40, thinning=10`

---

## 🔄 BayesPE → Textual Bayes Cheat Sheet

| BayesPE                             | Textual Bayes                                                |
| ----------------------------------- | ------------------------------------------------------------ |
| Multiple prompts                    | Single initial prompt                                        |
| `n_iterations_weights_optimiser=10` | `n_iterations_weights_optimiser=10` (maps to 100 MCMC steps) |
| Saves weights (.pkl)                | Saves prompts (.json)                                        |
| Weighted averaging                  | Discrete aggregation                                         |
| Fast training                       | Slower (LLM proposals)                                       |

---

## 🧪 Testing Commands

```bash
# Validate implementation
python bin/validate_textual_bayes.py

# Run Textual Bayes example
python bin/example_textual_bayes_zero_shot.py

# Compare both methods
python bin/compare_bayespe_vs_textual_bayes.py

# Inference only
python bin/textual_bayes_inference.py
```

---

## 📊 Output Differences

### BayesPE Output

```python
# Weighted probability distribution
[0.65, 0.25, 0.10]  # Weighted avg of p(y|x, prompt_i)
```

### Textual Bayes Output

```python
# Frequency-based confidence
[0.70, 0.20, 0.10]  # Count of discrete outputs / total
```

**Both sum to 1.0 and can be used identically for downstream tasks!**

---

## 🐛 Common Issues

### Issue: Low acceptance rate (<10%)

**Fix**: Reduce `batch_size` or adjust prior constraints

### Issue: All prompts identical

**Fix**: Increase `num_mcmc_steps` or try different `initial_prompt`

### Issue: Out of memory

**Fix**: Set `use_reduced_precision=True`

### Issue: "Textual Bayes not available"

**Fix**: Check that `textual_bayes.py` and `mhlp_sampler.py` exist in `src/`

---

## 💡 Pro Tips

1. **Start small**: Test with `num_mcmc_steps=20` first
2. **Monitor acceptance**: Should be 20-50% ideally
3. **Check diversity**: Use `get_prompt_diversity_stats()`
4. **Use priors**: Guide search with `prior_description`
5. **Save often**: MCMC can be interrupted and resumed

---

## 📈 Performance Expectations

| Dataset Size | MCMC Steps | Training Time | Quality |
| ------------ | ---------- | ------------- | ------- |
| 50-100       | 50         | ~30 min       | Good    |
| 100-500      | 100        | ~1-2 hrs      | Better  |
| 500+         | 200        | ~3-5 hrs      | Best    |

_Times vary based on LLM and hardware_

---

## 🎯 When to Use Textual Bayes

✅ **Use Textual Bayes when:**

- You want to discover better prompts automatically
- You have limited prompt engineering resources
- You need natural uncertainty quantification
- You're working on generative tasks
- You value exploration over speed

❌ **Stick with BayesPE when:**

- You have well-crafted prompt variations already
- You need fast training times
- You have limited compute budget
- You prefer deterministic results

---

## 📚 Documentation Hierarchy

1. **Quick Start**: This file (QUICK_REFERENCE.md)
2. **Full Guide**: TEXTUAL_BAYES_GUIDE.md
3. **Implementation**: IMPLEMENTATION_SUMMARY.md
4. **Code Examples**: bin/example\_\*.py
5. **Code Docs**: Inline comments in src/

---

## 🔗 Key Concepts

- **MHLP**: Metropolis-Hastings through LLM Proposals
- **Discrete Aggregation**: Counting outputs instead of averaging probabilities
- **Burn-in**: Initial MCMC samples to discard (convergence)
- **Thinning**: Reduce autocorrelation by keeping every nth sample
- **Prior**: Textual constraints to guide prompt search

---

## ✨ One-Liner Comparison

**BayesPE**: "Optimize weights over prompts I give you"
**Textual Bayes**: "Discover good prompts for me via MCMC"

---

## 🎓 Learning Path

1. Read this Quick Reference
2. Run `validate_textual_bayes.py`
3. Try `example_textual_bayes_zero_shot.py`
4. Compare with `compare_bayespe_vs_textual_bayes.py`
5. Read TEXTUAL_BAYES_GUIDE.md for depth
6. Adapt to your own task!

---

**Last Updated**: Implementation complete
**Status**: ✅ Production ready
