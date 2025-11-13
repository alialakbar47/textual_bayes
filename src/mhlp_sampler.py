"""
MHLP Sampler: Metropolis-Hastings through LLM Proposals
This module implements the core MCMC sampling algorithm for Textual Bayes.
"""

import random
import math
import numpy as np
import json
from typing import List, Tuple, Dict, Any
from llm_model import LLM


SMALL_CONSTANT = 0.00001


class MHLPSampler:
    """
    Metropolis-Hastings through LLM Proposals (MHLP) sampler for Textual Bayes.
    
    This sampler explores the space of prompts using MCMC to generate a posterior
    distribution of prompts that are effective for a given task.
    """
    
    def __init__(
        self,
        proposer_model_name: str = "gpt-4",
        evaluator_model_name: str = None,
        task_description: str = None,
        use_reduced_precision: bool = False,
        batch_size: int = 10
    ):
        """
        Initialize the MHLP sampler.
        
        :param proposer_model_name: LLM to use for proposing new prompts (should be powerful, e.g., GPT-4)
        :param evaluator_model_name: LLM to use for evaluating prompts on the task (can be same or different)
        :param task_description: Description of the task for the proposer
        :param use_reduced_precision: Whether to use reduced precision for models
        :param batch_size: Number of training examples to use per likelihood evaluation
        """
        # For proposal generation, we'll use the evaluator model to generate proposals
        # In a real implementation, you might want a separate API call to GPT-4
        if evaluator_model_name is None:
            evaluator_model_name = proposer_model_name
            
        self.evaluator_model = LLM(
            model_name=evaluator_model_name,
            use_reduced_precision=use_reduced_precision,
            temperature=0.7,
            max_tokens=500
        )
        self.task_description = task_description
        self.batch_size = batch_size
        
        # For prior evaluation, we use the same model
        self.prior_model = self.evaluator_model
        
    def run_mhlp(
        self,
        initial_prompt: str,
        train_data: List[Tuple[str, str]],
        num_steps: int = 100,
        prior_description: str = None,
        burn_in: int = 20,
        thinning: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Run the MHLP algorithm to generate a chain of prompts.
        
        :param initial_prompt: Starting prompt for the MCMC chain
        :param train_data: List of (input_text, true_output_text) tuples
        :param num_steps: Number of MCMC steps to run
        :param prior_description: Textual description of prompt constraints
        :param burn_in: Number of initial steps to discard
        :param thinning: Keep every nth sample after burn-in
        
        :return: Tuple of (sampled_prompts, acceptance_rates)
        """
        chain = [initial_prompt]
        acceptance_history = []
        current_prompt = initial_prompt
        
        # Calculate initial g(theta)
        log_g_current = self._calculate_log_g(
            current_prompt,
            train_data,
            prior_description
        )
        
        print(f"Starting MHLP sampling with {num_steps} steps...")
        print(f"Initial prompt: {current_prompt[:100]}...")
        print(f"Initial log g(theta): {log_g_current:.4f}\n")
        
        for t in range(num_steps):
            # 1. Propose a new prompt using LLM-based optimizer
            proposed_prompt = self._propose_new_prompt(current_prompt, train_data)
            
            # 2. Calculate the acceptance probability (in log space)
            log_g_proposed = self._calculate_log_g(
                proposed_prompt,
                train_data,
                prior_description
            )
            
            # Log acceptance ratio (assuming symmetric proposal distribution)
            log_acceptance_ratio = log_g_proposed - log_g_current
            acceptance_prob = min(1.0, math.exp(log_acceptance_ratio))
            
            # 3. Accept or reject the new prompt
            accepted = random.uniform(0, 1) < acceptance_prob
            
            if accepted:
                current_prompt = proposed_prompt
                log_g_current = log_g_proposed
                
            chain.append(current_prompt)
            acceptance_history.append(acceptance_prob)
            
            print(f"Step {t+1}/{num_steps}: "
                  f"Acceptance Prob = {acceptance_prob:.3f}, "
                  f"Accepted = {accepted}, "
                  f"log g = {log_g_current:.4f}")
            
        # Apply burn-in and thinning
        prompts_after_burn_in = chain[burn_in:]
        final_prompts = prompts_after_burn_in[::thinning]
        
        print(f"\nMCMC Sampling complete!")
        print(f"Total samples: {len(chain)}")
        print(f"After burn-in ({burn_in}): {len(prompts_after_burn_in)}")
        print(f"After thinning (every {thinning}): {len(final_prompts)}")
        print(f"Average acceptance rate: {np.mean(acceptance_history):.3f}")
        
        return final_prompts, acceptance_history
    
    def _calculate_log_g(
        self,
        prompt: str,
        data: List[Tuple[str, str]],
        prior_desc: str
    ) -> float:
        """
        Calculate log g(theta) = log p(D|theta) + log p(theta).
        
        :param prompt: The prompt to evaluate
        :param data: Training data
        :param prior_desc: Prior description
        
        :return: Log of unnormalized posterior
        """
        log_likelihood = self._log_p_D_given_theta(prompt, data)
        log_prior = self._log_p_theta(prompt, prior_desc) if prior_desc else 0.0
        
        return log_likelihood + log_prior
    
    def _propose_new_prompt(
        self,
        current_prompt: str,
        train_data: List[Tuple[str, str]]
    ) -> str:
        """
        Use an LLM to propose a new, improved prompt.
        
        :param current_prompt: Current prompt in the chain
        :param train_data: Training examples to show the proposer
        
        :return: Proposed new prompt
        """
        # Sample a few examples to show
        n_examples = min(3, len(train_data))
        sample_indices = random.sample(range(len(train_data)), n_examples)
        examples_str = "\n".join([
            f"Input: {train_data[i][0][:200]}\nExpected Output: {train_data[i][1]}"
            for i in sample_indices
        ])
        
        # Create meta-prompt for proposal
        meta_prompt = f"""You are an expert prompt engineer. Your task is to refine a given prompt to improve its performance on a specific task.

Based on the current prompt provided below, suggest a revised version. The new prompt should be more effective for {self.task_description if self.task_description else 'the task'}.

Here are some example inputs and expected outputs:
{examples_str}

Current Prompt:
---
{current_prompt}
---

Output only the new, revised prompt and nothing else. Do not add any commentary or preamble. The prompt should be clear, concise, and effective."""
        
        # Generate proposal using the model
        try:
            proposed_prompt = self.evaluator_model.generate_text(
                instructions="",
                content=meta_prompt
            )
            # Clean up the generated prompt
            proposed_prompt = proposed_prompt.strip()
            
            # If the model returns empty or very short, keep current
            if len(proposed_prompt) < 10:
                proposed_prompt = current_prompt
                
        except Exception as e:
            print(f"Error in proposal generation: {e}")
            proposed_prompt = current_prompt
            
        return proposed_prompt
    
    def _log_p_D_given_theta(
        self,
        prompt: str,
        data: List[Tuple[str, str]]
    ) -> float:
        """
        Calculate the log-likelihood of the data given a prompt.
        Uses a mini-batch from train_data for efficiency.
        
        :param prompt: The prompt to evaluate
        :param data: List of (input_text, true_output_text) tuples
        
        :return: Log-likelihood score
        """
        # Sample a mini-batch
        batch_size = min(self.batch_size, len(data))
        batch_indices = random.sample(range(len(data)), batch_size)
        data_batch = [data[i] for i in batch_indices]
        
        total_log_likelihood = 0.0
        
        for input_text, true_output_text in data_batch:
            # Combine prompt and input
            full_input = f"{prompt}\n\nInput: {input_text}\nOutput:"
            
            # Get log-probabilities for the true output
            # For now, we'll use a heuristic: generate and check if it matches
            # In a real implementation, you'd want token-level log-probs
            log_prob = self._get_logprobs_for_sequence(
                full_input,
                true_output_text
            )
            total_log_likelihood += log_prob
            
        # Normalize by batch size
        return total_log_likelihood / batch_size
    
    def _get_logprobs_for_sequence(
        self,
        prompt: str,
        target_sequence: str
    ) -> float:
        """
        Estimate log probability of generating target_sequence given prompt.
        
        This is a simplified implementation. In a full implementation, you would:
        1. Use the model's actual token-level log probabilities
        2. Sum log-probs for each token in target_sequence
        
        For now, we use a heuristic based on how well the model would generate
        something similar to the target.
        
        :param prompt: The full prompt to condition on
        :param target_sequence: The sequence we want to score
        
        :return: Estimated log probability
        """
        try:
            # Generate output from the model
            generated = self.evaluator_model.generate_text(
                instructions="",
                content=prompt
            ).strip()
            
            # Simple heuristic: measure similarity
            # In real implementation, use actual token log-probs
            
            # For classification tasks, check if the answer matches
            generated_lower = generated.lower().strip()
            target_lower = target_sequence.lower().strip()
            
            # Exact match gets high score
            if generated_lower == target_lower:
                return 0.0  # log(1) = 0
            
            # Partial match (contains the target)
            if target_lower in generated_lower or generated_lower in target_lower:
                return -0.5
            
            # Check if first few characters match (for longer outputs)
            overlap = sum(1 for a, b in zip(generated_lower, target_lower) if a == b)
            max_len = max(len(generated_lower), len(target_lower))
            
            if max_len > 0:
                similarity = overlap / max_len
                # Map similarity to log-prob (higher similarity = higher prob)
                return math.log(similarity + SMALL_CONSTANT)
            else:
                return -5.0  # Low log-prob for no match
                
        except Exception as e:
            print(f"Error in log-prob calculation: {e}")
            return -10.0  # Very low log-prob on error
    
    def _log_p_theta(
        self,
        prompt: str,
        prior_description: str
    ) -> float:
        """
        Calculate log prior probability of a prompt based on textual constraints.
        
        :param prompt: The prompt to evaluate
        :param prior_description: Textual description of desired prompt properties
        
        :return: Log prior probability
        """
        if prior_description is None:
            return 0.0  # Uniform prior
        
        # Meta-prompt for prior evaluation
        meta_prompt = f"""On a scale of 0.0 to 1.0, how well does the following PROMPT satisfy the given CONSTRAINT?
Respond ONLY with a JSON object in the format: {{"score": <value>}}

CONSTRAINT:
{prior_description}

PROMPT:
{prompt}
"""
        
        try:
            response = self.evaluator_model.generate_text(
                instructions="",
                content=meta_prompt
            ).strip()
            
            # Parse JSON response
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                score = float(result.get('score', 0.5))
            else:
                # Fallback: try to find a number in the response
                import re
                numbers = re.findall(r'0\.\d+|1\.0', response)
                score = float(numbers[0]) if numbers else 0.5
            
            # Clamp score to valid range
            score = max(0.01, min(0.99, score))
            
            return math.log(score)
            
        except Exception as e:
            print(f"Error in prior evaluation: {e}")
            return math.log(0.5)  # Neutral prior on error
    
    def save_chain(self, prompts: List[str], filepath: str):
        """
        Save the chain of prompts to a JSON file.
        
        :param prompts: List of prompts to save
        :param filepath: Path to save file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'prompts': prompts}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompts)} prompts to {filepath}")
    
    @staticmethod
    def load_chain(filepath: str) -> List[str]:
        """
        Load a chain of prompts from a JSON file.
        
        :param filepath: Path to load file
        
        :return: List of prompts
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompts = data['prompts']
        print(f"Loaded {len(prompts)} prompts from {filepath}")
        return prompts
