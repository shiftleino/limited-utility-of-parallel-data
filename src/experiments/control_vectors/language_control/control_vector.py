from ....training.olmo.tokenizer import Tokenizer
from ....training.olmo.model import OLMo
from ....training.olmo.config import ModelConfig
import torch
import os
import pandas as pd
from typing import List
import logging
import argparse
from .hooks import add_forward_hooks, get_extract_mean_activation_hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def calculate_direction(activations1: List[torch.Tensor], activations2: List[torch.Tensor]):
  """Calculate the control direction between two sets of activations by taking the mean of 
  the differences between the two sets of activations.

  Args:
      activations1 (List[torch.Tensor]): Activations of the first set of examples. Should be a list of tensors of shape (model_dim*num_layers).
      activations2 (List[torch.Tensor]): Activations of the second set of examples. Should be a list of tensors of shape (model_dim*num_layers).

  Returns:
      torch.Tensor: The mean of the differences between the two sets of activations for each residual stream dimension of each layer. 
      Shape is (1, model_dim*num_layers).
  """
  activations1 = torch.stack(activations1)
  activations2 = torch.stack(activations2)
  diff = activations2 - activations1
  return diff.mean(dim=0).unsqueeze(0)

def get_activations(model, tokenizer, examples: List[str]):
  """Get the activations of the model at each dimension of the residual stream of each layer for a list of examples.

  Args:
      model: The LLM used in the experiments for text generation.
      tokenizer: The tokenizer of the LLM.
      examples (List[str]): A list of example prompts for which the activations should be calculated.

  Returns:
      List[torch.Tensor]: A list of tensors of shape (model_dim*num_layers) containing the activations of the model for each example.
  """
  all_activations = []
  for i, example in enumerate(examples):
    activations = []
    module_hooks = [(layer, get_extract_mean_activation_hook(activations)) for layer in model.transformer.blocks]

    with torch.no_grad(), add_forward_hooks(module_hooks):
      input_ids = torch.tensor(tokenizer.encode(example, out_type=int), device=device).unsqueeze(0)
      _ = model(input_ids)

    activations_cached = [act.to("cuda:0") for act in activations]
    activations_all_layers = torch.cat(activations_cached, dim=0) # (num_layers * model_dim)
    all_activations.append(activations_all_layers)
  return all_activations

def get_control_direction(model, tokenizer, examples1: List[str], examples2: List[str]):
  """Calculates the control direction of the residual stream using contrastive examples.

  Args:
      model: The LLM used in the experiments for text generation.
      tokenizer: The tokenizer of the LLM
      examples1 (List[str]): List of examples used for calculating the control direction.
      examples2 (List[str]): The contrastive list of examples used for calculating the control direction.

  Returns:
      torch.Tensor: The residual stream control direction for all layers. Shape is (1, model_dim*num_layers).
  """
  logging.info("Extracting activations for examples1")
  activations1 = get_activations(model, tokenizer, examples1)
  logging.info("Extracting activations for examples2")
  activations2 = get_activations(model, tokenizer, examples2)
  logging.info("Calculating control direction")
  control_direction = calculate_direction(activations1, activations2)
  return control_direction

def main(args):
    tokenizer = Tokenizer(args.tokenizer_path)

    logging.info(f"Loading examples from files {args.filepath1} and {args.filepath2}")
    sentences1 = pd.read_parquet(args.filepath1)["text"].to_list()[:args.num_examples]
    sentences2 = pd.read_parquet(args.filepath2)["text"].to_list()[:args.num_examples]

    include_untrained = False
    if include_untrained:
        try:
            config = ModelConfig()
            model = OLMo(config)
            model.to(device)

            control_direction_mean = get_control_direction(model, tokenizer, sentences1, sentences2)
            logging.info("Saving the control vector to disk")
            torch.save(control_direction_mean, f"{args.output_dir}/control_vector_mean_{args.experiment_name}_untrained.pt")
    
        except ValueError as ve:
            logging.error(f"Skipping untrained model due to ValueError: {ve}")
        except Exception as e:
            logging.error(f"An error occurred while processing untrained model: {e}", exc_info=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing and cleaned up memory for untrained model")

    # Hardcoded steps for checkpoints due to laziness.
    start_step = 1000
    end_step = 1000
    checkpoint_paths = [args.checkpoints_dir + f"/step{i}" for i in range(start_step, end_step, 10000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    
    if not checkpoint_paths:
        logging.error(f"No checkpoint files found at {args.checkpoints_dir}. Please check the path.")
        return
    
    logging.info(f"Found {len(checkpoint_paths)} checkpoint files to process.")

    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        logging.info(f"Processing checkpoint ({checkpoint_idx+1}/{len(checkpoint_paths)}): {checkpoint_path}")
        try:
            model = OLMo.from_checkpoint(checkpoint_path, device=device)
            model.eval()
        except Exception as e:
            logging.error(f"Failed to load model from checkpoint {checkpoint_path}: {e}")
            continue

        try:
            control_direction_mean = get_control_direction(model, tokenizer, sentences1, sentences2)
            logging.info("Saving the control vector to disk")
            torch.save(control_direction_mean, f"{args.output_dir}/control_vector_mean_{args.experiment_name}_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.pt")

        except ValueError as ve:
            logging.error(f"Skipping checkpoint {checkpoint_path} due to ValueError: {ve}")
        except Exception as e:
            logging.error(f"An error occurred while processing checkpoint {checkpoint_path}: {e}", exc_info=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing and cleaned up memory for checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("No GPU available.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, help="The name of the experiment.")
    parser.add_argument("--filepath1", type=str, help="The path to the first file containing the examples.")
    parser.add_argument("--filepath2", type=str, help="The path to the second file containing the examples.")
    parser.add_argument("--num-examples", type=int, default=1000, help="The number of examples to use for the experiment.")
    parser.add_argument("--tokenizer-path", type=str, help="The path to the tokenizer model.")
    parser.add_argument("--checkpoints-dir", type=str, help="The path to the checkpoints of the model.")
    parser.add_argument("--last-step", type=str, default="47683", help="The last checkpoint step")
    parser.add_argument("--output-dir", type=str, default=".", help="The output directory for the results.")
    args = parser.parse_args()
    main(args)
