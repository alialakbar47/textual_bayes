import numpy as np
import pickle
from llm_model import LLM
from llm_classifier import LLMClassifier
from ensemble_scaler import EnsembleScaler

# Import new Textual Bayes implementation
# Set to True to use Textual Bayes (MCMC), False to use original BayesPE (weight optimization)
USE_TEXTUAL_BAYES = False

try:
    from textual_bayes import TextualBayes as TextualBayesImpl
    TEXTUAL_BAYES_AVAILABLE = True
except ImportError:
    TEXTUAL_BAYES_AVAILABLE = False
    print("Warning: Textual Bayes module not available. Using original BayesPE.")

SMALL_CONSTANT = 0.00001
LARGE_CONSTANT = 10000


def replace_nans_with_uniform(probs):
    """
    Function to replace probability distributions containing NaNs with uniform distributions (e.g. [NaN, 0.1] -> [0.5, 0.5])

    :param probs: 2D array of probability distributions [n_samples, n_classes]

    :return: 2D array of probability distributions with rows containing NaNs substituted with uniform [n_samples, n_classes]
    """
    for i in range(np.shape(probs)[0]):
        if np.isnan(probs[i,:]).any():
            probs[i, :] = np.divide(1.0, np.shape(probs)[1])*np.ones(np.shape(probs)[1])
    return probs


def smooth_probs_3d(probs_3d):
    """
    Function to format and smooth probability 3D arrays of probability distributions to avoid numerical precision errors.

    :param probs_3d: 3D array of probability distributions [n_samples, n_classes, n_instructions]

    :return: 3D array of probability distributions with rows containing NaNs substituted with uniform and smoothed out to avoid zeros [n_samples, n_classes, n_instructions]
    """
    probs_new = probs_3d + SMALL_CONSTANT
    for i in range(np.shape(probs_new)[2]):
        probs_new[:,:,i] = replace_nans_with_uniform(probs_new[:,:,i])
        for j in range(np.shape(probs_new)[0]):
            probs_new[j,:,i] = np.divide(probs_new[j,:,i], np.sum(probs_new[j,:,i]))
    return probs_new


class BayesPE(object):
    """
    Class for Bayesian Prompts Ensembles (BPE)
    
    Can operate in two modes:
    1. Original BayesPE: Optimizes weights over a fixed set of prompts
    2. Textual Bayes: Uses MCMC to explore the prompt space (set use_textual_bayes=True)
    """
    def __init__(self, model_name, prompt_formatting, instructions, few_shot_texts_sets=None, few_shot_labels_sets=None, max_len_content=None, use_reduced_precision=False, n_iterations_weights_optimiser=10, use_textual_bayes=None):
        """
        :param model_name: the Huggingface name of the LLM to use (e.g. 'mistralai/Mistral-7B-Instruct-v0.1')
        :param prompt_formatting: formatting script to retrieve classes names and schemas from
        :param instructions: list or 1D array of strings with the different task instructions to construct the BPE
        :param few_shot_texts_sets: lists of text examples to include in the prompt for few-shot operation.
               Should be a list of lists where each list contains the set of examples to include in the prompt for
               each of the instructions in 'instructions'.
        :param few_shot_labels_sets: 2D array with the labels corresponding to the text examples in 'few_shot_texts_sets'.
               size should be [n_few_shot_examples, n_instructions].
        :param max_len_content: maximum word count for content. content texts longer than this will be truncated. In None, no truncation is applied
        :param use_reduced_precision: whether to use reduced precision for the LLM to use less GPU memory and compute
        :param n_iterations_weights_optimiser: number of iterations for the weights optimiser
        :param use_textual_bayes: whether to use Textual Bayes (MCMC) mode instead of weight optimization
        """
        # Determine which mode to use
        if use_textual_bayes is None:
            use_textual_bayes = USE_TEXTUAL_BAYES and TEXTUAL_BAYES_AVAILABLE
        
        self.use_textual_bayes = use_textual_bayes
        
        if self.use_textual_bayes and TEXTUAL_BAYES_AVAILABLE:
            # Use Textual Bayes implementation
            print("Initializing in Textual Bayes mode (MCMC sampling)")
            self._textual_bayes = TextualBayesImpl(
                model_name=model_name,
                prompt_formatting=prompt_formatting,
                initial_prompt=instructions[0] if instructions else None,
                max_len_content=max_len_content,
                use_reduced_precision=use_reduced_precision,
                num_mcmc_steps=n_iterations_weights_optimiser * 10,
                burn_in=20,
                thinning=5
            )
            self.classifier = self._textual_bayes.classifier
            self.scaler = None
            self.instructions = instructions
            self.weights = None
            self.sampled_prompts = None
        else:
            # Use original BayesPE implementation
            if use_textual_bayes and not TEXTUAL_BAYES_AVAILABLE:
                print("Warning: Textual Bayes requested but not available. Using original BayesPE.")
            print("Initializing in BayesPE mode (weight optimization)")
            model = LLM(model_name=model_name, use_reduced_precision=use_reduced_precision)
            self.classifier = LLMClassifier(model, prompt_formatting, max_len_content=max_len_content)
            self.scaler = EnsembleScaler(n_iterations_weights_optimiser)
            self.instructions = instructions
            self.weights = np.divide(1.0, len(instructions))*np.ones(len(instructions))
            self._textual_bayes = None
            
        if few_shot_texts_sets is not None:
            self.examples_dict = self.make_few_shot_examples_dict(few_shot_texts_sets, few_shot_labels_sets)
        else:
            self.examples_dict = None

    def optimise_weights(self, input_texts, gt_labels, learning_rate=SMALL_CONSTANT):
        """
        :param input_texts: list or 1D array of strings with the validation text input examples
        :param gt_labels: list or 1D array of ground-truth labels corresponding to input_texts
        :param learning_rate: initial learning rate for the weights optimiser

        :return: 1D array of optimised weights (w^* in the paper) [n_instructions] or sampled prompts for Textual Bayes.
        """
        if self.use_textual_bayes:
            # Textual Bayes mode: run MCMC sampling
            print("Running MCMC sampling (Textual Bayes)...")
            self.sampled_prompts = self._textual_bayes.sample_prompts(input_texts, gt_labels)
            # Return dummy weights for compatibility
            self.weights = np.ones(len(self.sampled_prompts)) / len(self.sampled_prompts)
            return self.weights
        else:
            # Original BayesPE mode: optimize weights
            probs = self.classifier.sample_probs_ensemble(self.instructions, input_texts, examples_dict=self.examples_dict, n_samples=len(self.instructions))
            probs = smooth_probs_3d(probs)
            nan_cost = True
            lr = learning_rate
            while nan_cost:
                optimal_weights, costs = self.scaler.train(probs, gt_labels, lr=lr)
                if not np.isnan(costs[-1]):
                    nan_cost = False
                else:
                    lr = lr * 0.5
            self.weights = optimal_weights
            return optimal_weights

    def forward(self, input_texts, n_forward_passes=None):
        if self.use_textual_bayes:
            # Textual Bayes mode: discrete output aggregation
            return self._textual_bayes.forward(input_texts, n_forward_passes)
        else:
            # Original BayesPE mode: weighted ensemble
            if n_forward_passes is None:
                n_forward_passes = len(self.instructions)
            chosen_indices = np.argsort(self.weights)[-n_forward_passes:]
            chosen_weights = np.sort(self.weights)[-n_forward_passes:]
            probs = self.classifier.sample_probs_ensemble(self.instructions, input_texts, examples_dict=self.examples_dict, indices=chosen_indices)
            probs = smooth_probs_3d(probs)
            return self.scaler.scale_ensemble(probs, chosen_weights)

    def save_weights(self, save_dir='saved_weights/ensemble_weights'):
        if self.use_textual_bayes:
            # Save prompts instead of weights
            prompt_path = save_dir.replace('ensemble_weights', 'prompt_chain.json')
            self._textual_bayes.save_prompts(prompt_path)
        else:
            with open(save_dir, 'wb') as f:
                pickle.dump(self.weights, f)

    def load_weights(self, load_dir='saved_weights/ensemble_weights'):
        if self.use_textual_bayes:
            # Load prompts instead of weights
            prompt_path = load_dir.replace('ensemble_weights', 'prompt_chain.json')
            self._textual_bayes.load_prompts(prompt_path)
            self.sampled_prompts = self._textual_bayes.sampled_prompts
        else:
            with open(load_dir, 'rb') as f:
                self.weights = pickle.load(f)

    @staticmethod
    def make_few_shot_examples_dict(few_shot_texts_sets, few_shot_labels_sets):
        """
        :param few_shot_texts_sets: lists of text examples to include in the prompt for few-shot operation.
               Should be a list of lists where each list contains the set of examples to include in the prompt for
               each of the instructions in 'instructions'.
        :param few_shot_labels_sets: 2D array with the labels corresponding to the text examples in 'few_shot_texts_sets'.
               size should be [n_few_shot_examples, n_instructions].
        """
        examples_dict = {}
        for i in range(np.shape(few_shot_labels_sets)[1]):
            input_examples = few_shot_texts_sets[:, i]
            labels_examples = few_shot_labels_sets[:,i]
            examples_dict['input_examples_{}'.format(i)] = input_examples
            examples_dict['label_examples_{}'.format(i)] = labels_examples
        return examples_dict

    def print_prompt_example(self, index=0, input_text='<SAMPLE_IN>'):
        """
        Print out an example of the prompt that will be fed to the LLM with the current configuration.

        :param index: which prompt in the ensemble to print out an example for
        :param input_text: string with the input text to be evaluated.
        """
        if self.use_textual_bayes:
            self._textual_bayes.print_prompt_example(index, input_text)
        else:
            if self.examples_dict is None:
                input_examples = None
                labels_examples = None
            else:
                input_examples = self.examples_dict['input_examples_{}'.format(index)]
                labels_examples = self.examples_dict['label_examples_{}'.format(index)]

            self.classifier.print_prompt_example(self.instructions[index], input_text, input_examples=input_examples, labels_examples=labels_examples)



