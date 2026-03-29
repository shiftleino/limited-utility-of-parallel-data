import argparse
import logging
from typing import List
import os
import json
import torch
import random
from .pwcca import compute_pwcca
from ...training.olmo.model import OLMo
from ...training.olmo.config import ModelConfig
from ...training.olmo.tokenizer import Tokenizer
from .hooks import add_forward_hooks, get_extract_mean_activation_hook


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_activations(model: OLMo, examples: List[List[int]], device: str, n_layers: int = 24, model_dim: int = 2048):
    """Get the mean pooled representations of the model after each layer.

    Args:
        model (OLMo): The OLMo model instance.
        examples (List[List[int]]): A list of input tokens to process.
        device (str): The device to run the model on ('cuda' or 'cpu').
        n_layers (int, optional): Number of layers in the model. Defaults to 24.
        model_dim (int, optional): Dimension of the model's hidden states. Defaults to 2048.

    Returns:
        List[torch.Tensor]: A list of tensors, each of shape (model_dim, num_examples),
                            containing the mean pooled representations for each layer.
    """
    activations = [torch.zeros((model_dim, len(examples)), device=device) for _ in range(n_layers)]

    for i, example in enumerate(examples):
        activations_example = []
        module_hooks = [(layer, get_extract_mean_activation_hook(activations_example)) for layer in model.transformer.blocks]
        
        with torch.no_grad(), add_forward_hooks(module_hooks):
            input_ids = torch.tensor(example, device=device).unsqueeze(0)
            _ = model(input_ids)

        for layer_idx, activation in enumerate(activations_example):
            activations[layer_idx][:, i] = activation.squeeze().clone().detach()
    
    return activations

def main(args):
    if torch.cuda.is_available():
        device = args.device if args.device else "cuda"
    else:
        logging.warning("CUDA not available, using CPU. This will be very slow.")
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

    # Hardcode start and last steps for checkpoints because of laziness.
    start_step = 1000
    last_step = 1000

    checkpoint_files = [args.checkpoints_dir + f"/step{i}" for i in range(start_step, last_step, 2000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    
    if not checkpoint_files:
        logging.error(f"No checkpoint files found based on the provided directory and steps. Please check the path and parameters.")
        return
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint files to process.")

    finnish_sentences = []
    with open(args.finnish_data_path, "r") as f:
        for line in f:
            finnish_sentences.append(line.strip())
    print("Number of lines:", len(finnish_sentences))

    # Sample 20000 sentences
    indices = random.sample(range(len(finnish_sentences)), 20000)
    finnish_sentences = [finnish_sentences[i] for i in indices]
    logging.info("Sampled 20000 Finnish sentences for processing.")   
    
    english_sentences = []
    with open(args.english_data_path, "r") as f:
        for line in f:
            english_sentences.append(line.strip())
    print("Number of lines:", len(english_sentences))

    # Sample 20000 sentences
    english_sentences = [english_sentences[i] for i in indices]
    logging.info("Sampled 20000 English sentences for processing.")

    if len(finnish_sentences) != len(english_sentences):
        logging.error("The number of Finnish and English sentences do not match. Please check your data.")
        return
    
    finnish_tokens = []
    for sentence in finnish_sentences:
        tokens = tokenizer.encode(sentence, out_type=int)
        finnish_tokens.append(tokens)

    english_tokens = []
    for sentence in english_sentences:
        tokens = tokenizer.encode(sentence, out_type=int)
        english_tokens.append(tokens)

    all_results = []
    include_untrained_model = False
    if include_untrained_model:
        config = ModelConfig()
        model_untrained = OLMo(config)
        model_untrained.to(device)

        try:
            activations_finnish: List[torch.Tensor] = get_activations(
                model_untrained, finnish_tokens, device, n_layers=num_layers, model_dim=model_dim)
            
            activations_english: List[torch.Tensor] = get_activations(
                model_untrained, english_tokens, device, n_layers=num_layers, model_dim=model_dim)
            logging.info(f"Activations for untrained model calculated successfully.")
            
            all_pwcca_scores = []
            for layer_idx in range(num_layers):
                pwcca, _, _ = compute_pwcca(
                    activations_finnish[layer_idx].cpu().numpy(),
                    activations_english[layer_idx].cpu().numpy(),
                    epsilon=1e-5,
                )
                all_pwcca_scores.append(pwcca)
        
            all_results.append({
                "model": "untrained",
                "pwcca_scores": [str(item) for item in all_pwcca_scores],
            })

            output_json_path = os.path.join(args.output_dir, f"pwcca_experiment_{args.experiment_name}_step0.json")
            with open(output_json_path, 'w') as f:
                json.dump({
                "model": "untrained",
                "pwcca_scores": [str(item) for item in all_pwcca_scores],
            }, f, indent=4)

            logging.info(f"PWCCA scores for untrained model calculated successfully.")

        except ValueError as ve:
            logging.error(f"Skipping untrained model due to ValueError: {ve}")
        except Exception as e:
            logging.error(f"An error occurred while processing untrained model: {e}", exc_info=True)
        finally:
            del model_untrained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing and cleaned up memory for untrained model")

    
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_files):
        logging.info(f"Processing checkpoint ({checkpoint_idx+1}/{len(checkpoint_files)}): {checkpoint_path}")
        model_checkpoint = None
        try:
            model_checkpoint = OLMo.from_checkpoint(checkpoint_path, device=device)
            model_checkpoint.eval()
        except Exception as e:
            logging.error(f"Failed to load model from checkpoint {checkpoint_path}: {e}")
            continue

        try:
            activations_finnish = get_activations(
                model_checkpoint, finnish_tokens, device, n_layers=num_layers, model_dim=model_dim)
            activations_english = get_activations(
                model_checkpoint, english_tokens, device, n_layers=num_layers, model_dim=model_dim)
            logging.info(f"Activations for checkpoint {checkpoint_path} calculated successfully.")
            all_pwcca_scores = []
            for layer_idx in range(num_layers):
                pwcca, _, _ = compute_pwcca(
                    activations_finnish[layer_idx].cpu().numpy(),
                    activations_english[layer_idx].cpu().numpy(),
                )
                all_pwcca_scores.append(pwcca)
            
            checkpoint_name_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
            all_results.append({
                "model": checkpoint_name_stem,
                "pwcca_scores": [str(item) for item in all_pwcca_scores],
            })

            output_json_path = os.path.join(args.output_dir, f"pwcca_experiment_{args.experiment_name}_{checkpoint_name_stem}.json")
            with open(output_json_path, 'w') as f:
                json.dump({
                "model": checkpoint_name_stem,
                "pwcca_scores": [str(item) for item in all_pwcca_scores],
            }, f, indent=4)

            logging.info(f"PWCCA scores for checkpoint {checkpoint_path} calculated successfully.")
        
        except ValueError as ve:
            logging.error(f"Skipping checkpoint {checkpoint_path} due to ValueError: {ve}")
        except Exception as e:
            logging.error(f"An error occurred while processing checkpoint {checkpoint_path}: {e}", exc_info=True)
        finally:
            if model_checkpoint is not None:
                del model_checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing and cleaned up memory for checkpoint: {checkpoint_path}")
        
    logging.info("All checkpoints processed. Saving results...")
    output_json_path = os.path.join(args.output_dir, f"pwcca_experiment_{args.experiment_name}.json")
    print(all_results)
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model checkpoints to visualize and save PWCCA values.")
    
    parser.add_argument("--finnish_data_path", type=str, required=True, help="Path to the file containing Finnish examples.")
    parser.add_argument("--english_data_path", type=str, required=True, help="Path to the file containing English examples.")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing model checkpoint files.")
    parser.add_argument("--last_step", type=int, required=True, help="Last step number for checkpoints to process.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output figures.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment for output files.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Autodetects CUDA if None.")
    args = parser.parse_args()
        
    main(args)
