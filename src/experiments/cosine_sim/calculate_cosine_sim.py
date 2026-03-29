import argparse
import logging
from typing import List
import os
import json
import torch
import random
import torch.nn.functional as F
from ...training.olmo.model import OLMo
from ...training.olmo.config import ModelConfig
from ...training.olmo.tokenizer import Tokenizer
from .hooks import add_forward_hooks, get_extract_mean_activation_hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_activations(model: OLMo, examples: List[List[int]], device: str, n_layers: int = 24, model_dim: int = 2048) -> List[torch.Tensor]:
    """Get the mean pooled representations of the model after each layer.

    Args:
        model: The language model instance.
        examples (List[List[int]]): A list of input tokens to process.
        device (str): The device to run the model on ('cuda' or 'cpu').
        n_layers (int, optional): Number of layers in the model. Defaults to 24.
        model_dim (int, optional): Dimension of the model's hidden states. Defaults to 2048.

    Returns:
        List[torch.Tensor]: A list of tensors, each of shape (num_examples, model_dim),
                            containing the mean pooled representations for each layer.
    """
    activations = [torch.zeros((len(examples), model_dim), device=device) for _ in range(n_layers)]
    for i, example in enumerate(examples):
        activations_example = []
        module_hooks = [(layer, get_extract_mean_activation_hook(activations_example)) for layer in model.transformer.blocks]
        
        with torch.no_grad(), add_forward_hooks(module_hooks):
            input_ids = torch.tensor(example, device=device).unsqueeze(0)
            _ = model(input_ids)

        for layer_idx, activation in enumerate(activations_example):
            activations[layer_idx][i, :] = activation.squeeze().clone().detach()
    
    return activations

def compute_cosine_similarity(activations1: torch.Tensor, activations2: torch.Tensor) -> float:
    """Computes the mean cosine similarity between two sets of activation tensors.

    Args:
        activations1 (torch.Tensor): The first set of activations (num_examples, model_dim).
        activations2 (torch.Tensor): The second set of activations (num_examples, model_dim).

    Returns:
        float: The mean cosine similarity score.
    """
    cosine_sim = F.cosine_similarity(activations1, activations2, dim=1)
    return cosine_sim.mean().item()

def main(args):
    if torch.cuda.is_available() and args.device != 'cpu':
        device = args.device if args.device else "cuda"
    else:
        if args.device == 'cuda':
            logging.warning("CUDA not available, falling back to CPU. This will be very slow.")
        device = "cpu"
    logging.info(f"Using device: {device}")

    logging.info(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        tokenizer = Tokenizer(args.tokenizer_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return
    
    num_layers = 24
    model_dim = 2048

    # Hard coded start and end steps for checkpoints because of laziness.
    start_step = 1000
    end_step = 1000

    checkpoint_files = [args.checkpoints_dir + f"/step{i}" for i in range(start_step, end_step, 1000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    
    if not checkpoint_files:
        logging.error("No checkpoint files found. Please check the directory and parameters.")
        return
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint files to process.")

    finnish_sentences = []
    with open(args.finnish_data_path, "r") as f:
        for line in f:
            finnish_sentences.append(line.strip())
    print("Number of lines:", len(finnish_sentences))

    # Sample 2000 sentences
    indices = random.sample(range(len(finnish_sentences)), 2000)

    finnish_sentences = [finnish_sentences[i] for i in indices]
    logging.info("Sampled 2000 Finnish sentences for processing.")   
    
    english_sentences = []
    with open(args.english_data_path, "r") as f:
        for line in f:
            english_sentences.append(line.strip())
    print("Number of lines:", len(english_sentences))

    english_sentences = [english_sentences[i] for i in indices]
    logging.info("Sampled 2000 English sentences for processing.")

    if len(finnish_sentences) != len(english_sentences):
        logging.error("The number of Finnish and English sentences do not match.")
        return
    
    finnish_tokens = [tokenizer.encode(s, out_type=int) for s in finnish_sentences]
    english_tokens = [tokenizer.encode(s, out_type=int) for s in english_sentences]

    all_results = []
    
    include_untrained_model = True
    if include_untrained_model:
        logging.info("Processing untrained model...")
        config = ModelConfig()
        model_untrained = OLMo(config).to(device)

        try:
            activations_fi = get_activations(model_untrained, finnish_tokens, device, num_layers, model_dim)
            activations_en = get_activations(model_untrained, english_tokens, device, num_layers, model_dim)
            
            translation_scores = []
            random_scores = []

            n_samples = len(finnish_tokens)
            n_random_samples = 50

            for i in range(num_layers):
                translation_scores.append(compute_cosine_similarity(activations_fi[i], activations_en[i]))

                per_sentence_random_similarities = torch.zeros((n_samples, n_random_samples), device=device)

                for n in range(n_random_samples):
                    shuffled_indices = torch.randperm(n_samples)
                    shuffled_en_activations = activations_en[i][shuffled_indices]
                    per_sentence_random_similarities[:, n] = F.cosine_similarity(activations_fi[i], shuffled_en_activations, dim=1)
                
                random_sim = per_sentence_random_similarities.mean(dim=1)
                random_scores.append(random_sim.mean().item())

            result_data = {
                "model": "untrained",
                "translation_cosine_similarities": [str(s) for s in translation_scores],
                "random_cosine_similarities": [str(s) for s in random_scores],
                "difference": [str(t - r) for t, r in zip(translation_scores, random_scores)],
            }
            all_results.append(result_data)

            output_json_path = os.path.join(args.output_dir, f"cosine_similarity_{args.experiment_name}_untrained.json")
            with open(output_json_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            logging.info(f"Results for untrained model saved to {output_json_path}")

        except Exception as e:
            logging.error(f"An error occurred while processing the untrained model: {e}", exc_info=True)
        finally:
            del model_untrained, activations_fi, activations_en
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Finished processing and cleaned up memory for untrained model.")

    for idx, checkpoint_path in enumerate(checkpoint_files):
        logging.info(f"Processing checkpoint ({idx+1}/{len(checkpoint_files)}): {checkpoint_path}")
        model_checkpoint = None
        try:
            model_checkpoint = OLMo.from_checkpoint(checkpoint_path, device=device)
            model_checkpoint.eval()

            activations_fi = get_activations(model_checkpoint, finnish_tokens, device, num_layers, model_dim)
            activations_en = get_activations(model_checkpoint, english_tokens, device, num_layers, model_dim)

            translation_scores = []
            random_scores = []
            
            n_samples = len(finnish_tokens)
            n_random_samples = 50

            for i in range(num_layers):
                translation_scores.append(compute_cosine_similarity(activations_fi[i], activations_en[i]))

                per_sentence_random_similarities = torch.zeros((n_samples, n_random_samples), device=device)

                for n in range(n_random_samples):
                    shuffled_indices = torch.randperm(n_samples)
                    shuffled_en_activations = activations_en[i][shuffled_indices]
                    per_sentence_random_similarities[:, n] = F.cosine_similarity(activations_fi[i], shuffled_en_activations, dim=1)
                
                random_sim = per_sentence_random_similarities.mean(dim=1)
                random_scores.append(random_sim.mean().item())

            checkpoint_name = os.path.basename(checkpoint_path)
            result_data = {
                "model": checkpoint_name,
                "translation_cosine_similarities": [str(s) for s in translation_scores],
                "random_cosine_similarities": [str(s) for s in random_scores],
                "difference": [str(t - r) for t, r in zip(translation_scores, random_scores)],
            }
            all_results.append(result_data)
            
            output_json_path = os.path.join(args.output_dir, f"cosine_similarity_{args.experiment_name}_{checkpoint_name}.json")
            with open(output_json_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            logging.info(f"Results for {checkpoint_name} saved to {output_json_path}")

        except Exception as e:
            logging.error(f"Failed processing checkpoint {checkpoint_path}: {e}", exc_info=True)
        finally:
            if model_checkpoint is not None:
                del model_checkpoint, activations_fi, activations_en
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing and cleaned up memory for {checkpoint_path}")
        
    logging.info("All checkpoints processed. Saving combined results...")
    final_output_path = os.path.join(args.output_dir, f"cosine_similarity_experiment_{args.experiment_name}_{start_step}-{args.last_step}.json")
    with open(final_output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"All results saved to {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--finnish_data_path", type=str, required=True, help="Path to the Parquet file containing Finnish sentences.")
    parser.add_argument("--english_data_path", type=str, required=True, help="Path to the Parquet file containing English sentences.")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing model checkpoint files.")
    parser.add_argument("--last_step", type=int, default=47000, help="Last step number for checkpoints. Default is 47000.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSON files and plots.")
    parser.add_argument("--experiment_name", type=str, required=True, help="A unique name for the experiment to label output files.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Autodetects CUDA if not specified.")
    
    args = parser.parse_args()
        
    main(args)
