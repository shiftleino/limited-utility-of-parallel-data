import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ...training.olmo.tokenizer import Tokenizer
from ...training.olmo.model import OLMo
from ...training.olmo.config import ModelConfig
from .hooks import add_forward_hooks, get_extract_mean_activation_hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_activations(model: OLMo, tokenizer: Tokenizer, examples: List[str], device: str):
  """Get the activations of the model at each dimension of the residual stream of each layer for a list of examples.

  Args:
      model: The LLM used in the experiments for text generation.
      tokenizer: The tokenizer of the LLM.
      examples (List[str]): A list of example prompts for which the activations should be calculated.
      device (str): The device to run the model and store tensors on.

  Returns:
      List[torch.Tensor]: A list of tensors, each of shape (num_layers * model_dim),
                          containing the concatenated mean activations for all layers for each example.
  """
  all_activations = []
  for i, example in enumerate(examples):
    activations_for_example = []
    
    module_hooks = [(layer, get_extract_mean_activation_hook(activations_for_example)) for layer in model.transformer.blocks]

    with torch.no_grad(), add_forward_hooks(module_hooks):
      input_ids = torch.tensor(tokenizer.encode(example, out_type=int), device=device).unsqueeze(0)
      _ = model(input_ids)

    activations_all_layers = torch.cat(activations_for_example, dim=0)
    all_activations.append(activations_all_layers)
  return all_activations

def create_activations_for_viz(model, tokenizer, finnish_data_path: str, english_data_path: str, num_examples: int, device: str):
    """
    Loads Finnish and English sentences, gets their activations from the model,
    and returns them as stacked tensors.
    """
    logging.info(f"Loading examples from files: {finnish_data_path}, {english_data_path}")
    try:
        finnish_sentences = pd.read_parquet(finnish_data_path)["text"].to_list()[:num_examples]
        english_sentences = pd.read_parquet(english_data_path)["text"].to_list()[:num_examples]
    except Exception as e:
        logging.error(f"Error loading or parsing Parquet files: {e}")
        raise

    logging.info(f"Extracting activations for {len(finnish_sentences)} Finnish sentences...")
    activations_fi_list = get_activations(model, tokenizer, finnish_sentences, device)
    
    logging.info(f"Extracting activations for {len(english_sentences)} English sentences...")
    activations_en_list = get_activations(model, tokenizer, english_sentences, device)

    if not activations_fi_list or not activations_en_list:
        logging.error("No activations were generated. Check input data and model.")
        raise ValueError("Activation generation failed for one or both languages.")

    activations_finnish_stacked = torch.stack(activations_fi_list)
    activations_english_stacked = torch.stack(activations_en_list)

    return activations_finnish_stacked, activations_english_stacked

def viz_layers_activations_lang(
    activations1: torch.Tensor, 
    activations2: torch.Tensor, 
    num_layers: int, 
    model_hidden_dim: int, 
    base_save_path: str,
    step: str
):
    """
    Visualizes PCA-reduced activations.
    """
    num_subplot_rows = 6
    num_subplot_cols = 4
    
    fig_pca, axes_pca = plt.subplots(num_subplot_rows, num_subplot_cols, figsize=(24, num_subplot_rows * 5), constrained_layout=True)
    axes_pca = axes_pca.ravel()
    logging.info("Generating PCA visualizations...")

    for i in range(num_layers):
        start_dim_idx = i * model_hidden_dim
        end_dim_idx = (i + 1) * model_hidden_dim

        segment1_data = activations1.cpu().numpy()[:, start_dim_idx:end_dim_idx]
        segment2_data = activations2.cpu().numpy()[:, start_dim_idx:end_dim_idx]
        
        num_samples1 = segment1_data.shape[0]
        combined_segment_data = np.vstack((segment1_data, segment2_data))

        pca = PCA(n_components=2, random_state=42)
        transformed_data_pca = pca.fit_transform(combined_segment_data)

        transformed_segment1_pca = transformed_data_pca[:num_samples1, :]
        transformed_segment2_pca = transformed_data_pca[num_samples1:, :]
        
        if i < len(axes_pca):
            ax = axes_pca[i]
            ax.scatter(transformed_segment1_pca[:, 0], transformed_segment1_pca[:, 1], label='Finnish', alpha=0.7, s=10, color='blue')
            ax.scatter(transformed_segment2_pca[:, 0], transformed_segment2_pca[:, 1], label='English', alpha=0.7, s=10, color='red')
            ax.set_title(f'Layer {i+1} (PCA)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.legend()
        else:
            logging.warning(f"PCA: Attempted to access axis index {i} beyond available axes ({len(axes_pca)}). This might happen if num_layers is odd and subplot layout is not perfectly matched.")

    fig_pca.suptitle(f'PCA of Activations for {step} checkpoint', fontsize=16)

    pca_save_path = f"{os.path.splitext(base_save_path)[0]}_pca.png"
    try:
        plt.savefig(pca_save_path)
        logging.info(f"Saved PCA visualization to {pca_save_path}")
    except Exception as e:
        logging.error(f"Failed to save PCA figure to {pca_save_path}: {e}")
    plt.close(fig_pca)


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
    
    # Hardcode start and last steps for checkpoints because of laziness.
    start_step = 1000
    last_step = 1000

    checkpoint_files = [args.checkpoints_dir + f"/step{i}" for i in range(start_step, last_step, 1000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    
    if not checkpoint_files:
        logging.error(f"No checkpoint files found based on the provided directory and steps. Please check the path and parameters.")
        return
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint files to process.")

    config = ModelConfig()
    model_untrained = OLMo(config)
    model_untrained.to(device)

    try:
        num_layers = len(model_untrained.transformer.blocks)
        model_hidden_dim = model_untrained.config.d_model

        activations_finnish, activations_english = create_activations_for_viz(
            model_untrained, tokenizer, args.finnish_data_path, args.english_data_path,
            args.num_examples, device
        )
        
        base_figure_name_untrained = f"viz_activations_{args.experiment_name}_untrained"
        base_figure_save_path_untrained = os.path.join(args.figures_output_dir, base_figure_name_untrained)

        logging.info(f"Visualizing activations for untrained model...")
        viz_layers_activations_lang(
            activations_finnish, activations_english,
            num_layers, model_hidden_dim,
            base_save_path=base_figure_save_path_untrained,
            step="untrained"
        )
    
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
            num_layers = len(model_checkpoint.transformer.blocks)
            model_hidden_dim = model_checkpoint.config.d_model

            activations_finnish, activations_english = create_activations_for_viz(
                model_checkpoint, tokenizer, args.finnish_data_path, args.english_data_path,
                args.num_examples, device
            )

            checkpoint_name_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
            base_figure_name_checkpoint = f"viz_activations_{args.experiment_name}_{checkpoint_name_stem}"
            base_figure_save_path_checkpoint = os.path.join(args.figures_output_dir, base_figure_name_checkpoint)

            logging.info(f"Visualizing activations for {checkpoint_path}...")
            viz_layers_activations_lang(
                activations_finnish, activations_english,
                num_layers, model_hidden_dim,
                base_save_path=base_figure_save_path_checkpoint,
                step=checkpoint_name_stem
            )
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model checkpoints to visualize and save language-specific activations.")
    
    parser.add_argument("--finnish_data_path", type=str, required=True, help="Path to the Parquet file containing Finnish examples.")
    parser.add_argument("--english_data_path", type=str, required=True, help="Path to the Parquet file containing English examples.")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing model checkpoint files.")
    parser.add_argument("--last_step", type=int, required=True, help="Last step number for checkpoints to process.")

    parser.add_argument("--figures_output_dir", type=str, required=True, help="Directory to save the output figures.")
    
    parser.add_argument("--experiment_name", type=str, default="default_exp", help="Name of the experiment for output file naming.")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to use from each language file.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Autodetects CUDA if None.")

    args = parser.parse_args()
        
    main(args)
