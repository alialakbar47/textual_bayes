"""
Textual Bayes: Bayesian Inference over the Space of Prompts

This module implements the Textual Bayes methodology, which uses MCMC sampling
to explore the prompt space and generate uncertainty estimates through discrete
output aggregation.
"""

import numpy as np
import pickle
import json
from collections import Counter
from typing import List, Tuple, Dict, Any
from llm_model import LLM
from llm_classifier import LLMClassifier
from mhlp_sampler import MHLPSampler

SMALL_CONSTANT = 0.00001


class TextualBayes:
    """
    Textual Bayes class for Bayesian inference over prompts.
    
    Unlike BayesPE which optimizes weights over a fixed set of prompts,
    Textual Bayes uses MCMC to explore the space of prompts and generates
    predictions through discrete output aggregation.
    """
    
    def __init__(
        self,
        model_name: str,
        prompt_formatting,
        initial_prompt: str = None,
        task_description: str = None,
        max_len_content: int = None,
        use_reduced_precision: bool = False,
        num_mcmc_steps: int = 100,
        burn_in: int = 20,
        thinning: int = 5
    ):
        """
        Initialize Textual Bayes.
        
        :param model_name: Huggingface name of the LLM to use
        :param prompt_formatting: Formatting script with classes and schemas
        :param initial_prompt: Starting prompt for MCMC (if None, uses default from prompt_formatting)
        :param task_description: Description of the task for prompt proposal
        :param max_len_content: Maximum word count for content truncation
        :param use_reduced_precision: Whether to use reduced precision
        :param num_mcmc_steps: Number of MCMC sampling steps
        :param burn_in: Number of initial MCMC steps to discard
        :param thinning: Keep every nth sample after burn-in
        """
        # Initialize the LLM and classifier
        model = LLM(model_name=model_name, use_reduced_precision=use_reduced_precision)
        self.classifier = LLMClassifier(model, prompt_formatting, max_len_content=max_len_content)
        
        # Initialize MHLP sampler
        self.sampler = MHLPSampler(
            proposer_model_name=model_name,
            evaluator_model_name=model_name,
            task_description=task_description,
            use_reduced_precision=use_reduced_precision,
            batch_size=10
        )
        
        # Store prompt formatting info
        self.classes_strings = prompt_formatting.CLASSES
        self.format_instruction = prompt_formatting.format_instruction
        
        # MCMC parameters
        self.num_mcmc_steps = num_mcmc_steps
        self.burn_in = burn_in
        self.thinning = thinning
        
        # Initialize with a single prompt
        if initial_prompt is None:
            self.initial_prompt = prompt_formatting.INSTRUCTION
        else:
            self.initial_prompt = initial_prompt
            
        # This will store the sampled prompts after training
        self.sampled_prompts = None
        
    def sample_prompts(
        self,
        input_texts: List[str],
        gt_labels: List[int],
        prior_description: str = None
    ) -> List[str]:
        """
        Run MCMC sampling to generate a posterior distribution of prompts.
        This replaces the weight optimization in BayesPE.
        
        :param input_texts: Training text inputs
        :param gt_labels: Ground-truth labels for training inputs
        :param prior_description: Textual description of prompt constraints
        
        :return: List of sampled prompts after burn-in and thinning
        """
        # Prepare training data as (input, output) pairs
        train_data = []
        for i, (text, label) in enumerate(zip(input_texts, gt_labels)):
            output_text = self.classes_strings[label]
            train_data.append((text, output_text))
        
        # Run MHLP sampling
        sampled_prompts, acceptance_history = self.sampler.run_mhlp(
            initial_prompt=self.initial_prompt,
            train_data=train_data,
            num_steps=self.num_mcmc_steps,
            prior_description=prior_description,
            burn_in=self.burn_in,
            thinning=self.thinning
        )
        
        self.sampled_prompts = sampled_prompts
        return sampled_prompts
    
    def predict_with_uncertainty(
        self,
        input_texts: List[str],
        sampled_prompts: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty using discrete output aggregation.
        
        Instead of weighted averaging of probabilities (BayesPE), we:
        1. Generate discrete outputs for each sampled prompt
        2. Aggregate these outputs to compute confidence distributions
        
        :param input_texts: Test inputs to predict on
        :param sampled_prompts: Prompts to use (if None, uses self.sampled_prompts)
        
        :return: Tuple of (predicted_labels, confidence_distributions)
                 - predicted_labels: shape [n_samples]
                 - confidence_distributions: shape [n_samples, n_classes]
        """
        if sampled_prompts is None:
            if self.sampled_prompts is None:
                raise ValueError("No sampled prompts available. Run sample_prompts() first.")
            sampled_prompts = self.sampled_prompts
        
        n_samples = len(input_texts)
        n_classes = len(self.classes_strings)
        
        # Store all discrete outputs
        all_outputs = []
        
        print(f"Generating predictions using {len(sampled_prompts)} sampled prompts...")
        
        for i, input_text in enumerate(input_texts):
            outputs_for_sample = []
            
            for prompt_idx, prompt in enumerate(sampled_prompts):
                # Generate discrete output for this prompt
                output = self._generate_discrete_output(prompt, input_text)
                outputs_for_sample.append(output)
            
            all_outputs.append(outputs_for_sample)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_samples} samples")
        
        # Compute confidence distributions from discrete outputs
        predicted_labels = np.zeros(n_samples, dtype=int)
        confidence_distributions = np.zeros((n_samples, n_classes))
        
        for i, outputs in enumerate(all_outputs):
            # Count frequency of each output
            output_counts = Counter(outputs)
            total = len(outputs)
            
            # Convert to probability distribution
            for label, count in output_counts.items():
                if label >= 0 and label < n_classes:  # Valid label
                    confidence_distributions[i, label] = count / total
            
            # Predicted label is the most frequent
            predicted_labels[i] = max(output_counts.items(), key=lambda x: x[1])[0]
        
        return predicted_labels, confidence_distributions
    
    def _generate_discrete_output(self, prompt: str, input_text: str) -> int:
        """
        Generate a single discrete output (class label) for an input.
        
        :param prompt: The prompt to use
        :param input_text: Input text to classify
        
        :return: Predicted class label (integer)
        """
        # Get class probabilities
        probs = self.classifier.soft_label(
            instruction=prompt,
            input_text=input_text
        )
        
        # Sample from the distribution (or take argmax for deterministic)
        # For more diversity, we sample; for consistency, we could use argmax
        predicted_label = np.argmax(probs)
        
        return predicted_label
    
    def forward(
        self,
        input_texts: List[str],
        n_forward_passes: int = None
    ) -> np.ndarray:
        """
        Generate predictions using the sampled prompts.
        This method maintains compatibility with BayesPE interface.
        
        :param input_texts: Test inputs
        :param n_forward_passes: Number of prompts to use (if None, uses all)
        
        :return: Confidence distributions [n_samples, n_classes]
        """
        if self.sampled_prompts is None:
            raise ValueError("No sampled prompts available. Run sample_prompts() first.")
        
        # Select subset of prompts if requested
        if n_forward_passes is not None and n_forward_passes < len(self.sampled_prompts):
            selected_prompts = self.sampled_prompts[:n_forward_passes]
        else:
            selected_prompts = self.sampled_prompts
        
        _, confidence_distributions = self.predict_with_uncertainty(
            input_texts,
            sampled_prompts=selected_prompts
        )
        
        return confidence_distributions
    
    def save_prompts(self, save_dir: str = 'saved_prompts/prompt_chain.json'):
        """
        Save the sampled prompts to a file.
        
        :param save_dir: Path to save the prompts
        """
        if self.sampled_prompts is None:
            raise ValueError("No sampled prompts to save. Run sample_prompts() first.")
        
        self.sampler.save_chain(self.sampled_prompts, save_dir)
    
    def load_prompts(self, load_dir: str = 'saved_prompts/prompt_chain.json'):
        """
        Load sampled prompts from a file.
        
        :param load_dir: Path to load the prompts from
        """
        self.sampled_prompts = self.sampler.load_chain(load_dir)
    
    def print_prompt_example(self, index: int = 0, input_text: str = '<SAMPLE_IN>'):
        """
        Print an example of a prompt from the sampled chain.
        
        :param index: Which prompt to print (0 = initial, or index into sampled_prompts)
        :param input_text: Sample input text
        """
        if self.sampled_prompts is not None and index > 0:
            # Adjust index to be within sampled prompts
            prompt_idx = min(index - 1, len(self.sampled_prompts) - 1)
            prompt = self.sampled_prompts[prompt_idx]
        else:
            prompt = self.initial_prompt
        
        self.classifier.print_prompt_example(
            instruction=prompt,
            input_text=input_text
        )
    
    def get_prompt_diversity_stats(self) -> Dict[str, Any]:
        """
        Compute statistics about the diversity of sampled prompts.
        
        :return: Dictionary with diversity metrics
        """
        if self.sampled_prompts is None:
            return {"error": "No sampled prompts available"}
        
        # Number of unique prompts
        unique_prompts = len(set(self.sampled_prompts))
        
        # Most common prompt
        prompt_counts = Counter(self.sampled_prompts)
        most_common_prompt, max_count = prompt_counts.most_common(1)[0]
        
        # Average prompt length
        avg_length = np.mean([len(p) for p in self.sampled_prompts])
        
        return {
            "total_prompts": len(self.sampled_prompts),
            "unique_prompts": unique_prompts,
            "diversity_ratio": unique_prompts / len(self.sampled_prompts),
            "most_common_count": max_count,
            "most_common_prompt": most_common_prompt[:200],
            "average_prompt_length": avg_length
        }


# Backward compatibility: alias TextualBayes as BayesPE for easy transition
class BayesPE(TextualBayes):
    """
    Backward compatibility wrapper.
    Maps old BayesPE API to new TextualBayes implementation.
    """
    
    def __init__(
        self,
        model_name: str,
        prompt_formatting,
        instructions: List[str] = None,
        few_shot_texts_sets=None,
        few_shot_labels_sets=None,
        max_len_content: int = None,
        use_reduced_precision: bool = False,
        n_iterations_weights_optimiser: int = 10
    ):
        """
        Initialize with BayesPE-compatible interface.
        
        Note: instructions parameter is deprecated. We use the first instruction
        as the initial_prompt for MCMC.
        """
        # Use first instruction as initial prompt, or default
        if instructions is not None and len(instructions) > 0:
            initial_prompt = instructions[0]
        else:
            initial_prompt = None
        
        # Map n_iterations to num_mcmc_steps
        num_mcmc_steps = max(n_iterations_weights_optimiser * 10, 100)
        
        super().__init__(
            model_name=model_name,
            prompt_formatting=prompt_formatting,
            initial_prompt=initial_prompt,
            max_len_content=max_len_content,
            use_reduced_precision=use_reduced_precision,
            num_mcmc_steps=num_mcmc_steps,
            burn_in=20,
            thinning=5
        )
        
        # Store for compatibility (though not used)
        self.instructions = instructions if instructions is not None else [initial_prompt]
        
        # Few-shot support would require additional implementation
        if few_shot_texts_sets is not None:
            print("Warning: Few-shot learning not yet fully implemented in Textual Bayes")
            self.examples_dict = self.make_few_shot_examples_dict(
                few_shot_texts_sets,
                few_shot_labels_sets
            )
        else:
            self.examples_dict = None
        
        # For compatibility
        self.weights = None
    
    def optimise_weights(
        self,
        input_texts: List[str],
        gt_labels: List[int],
        learning_rate: float = SMALL_CONSTANT
    ) -> np.ndarray:
        """
        Compatibility method: runs MCMC sampling instead of weight optimization.
        
        :param input_texts: Validation text inputs
        :param gt_labels: Ground-truth labels
        :param learning_rate: (Ignored in Textual Bayes)
        
        :return: Dummy weights array for compatibility
        """
        print("Running MCMC sampling (Textual Bayes mode)...")
        self.sample_prompts(input_texts, gt_labels)
        
        # Return dummy weights for compatibility
        self.weights = np.ones(len(self.sampled_prompts)) / len(self.sampled_prompts)
        return self.weights
    
    def save_weights(self, save_dir: str = 'saved_weights/ensemble_weights'):
        """Compatibility method: saves prompts instead of weights."""
        # Change extension to .json
        if save_dir.endswith('ensemble_weights'):
            save_dir = save_dir.replace('ensemble_weights', 'prompt_chain.json')
        self.save_prompts(save_dir)
    
    def load_weights(self, load_dir: str = 'saved_weights/ensemble_weights'):
        """Compatibility method: loads prompts instead of weights."""
        # Change extension to .json
        if load_dir.endswith('ensemble_weights'):
            load_dir = load_dir.replace('ensemble_weights', 'prompt_chain.json')
        self.load_prompts(load_dir)
    
    @staticmethod
    def make_few_shot_examples_dict(few_shot_texts_sets, few_shot_labels_sets):
        """Compatibility method for few-shot examples."""
        examples_dict = {}
        for i in range(np.shape(few_shot_labels_sets)[1]):
            input_examples = few_shot_texts_sets[:, i]
            labels_examples = few_shot_labels_sets[:, i]
            examples_dict[f'input_examples_{i}'] = input_examples
            examples_dict[f'label_examples_{i}'] = labels_examples
        return examples_dict
