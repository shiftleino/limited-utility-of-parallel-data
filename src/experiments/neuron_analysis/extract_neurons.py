from typing import List
import torch
import argparse
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from ...training.olmo.model import OLMo
from ...training.olmo.tokenizer import Tokenizer
from ...training.olmo.config import ModelConfig
from .hooks import get_extract_mean_activation_hook, add_forward_hooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_activations(model: OLMo, tokenizer: Tokenizer, examples: List[str], device: str):
  """Get the mean activations of the model at each neuron for each layer.

  Args:
      model: The LLM used in the experiments for text generation.
      tokenizer: The tokenizer of the LLM.
      examples (List[str]): A list of example prompts for which the activations should be calculated.
      device (str): The device to run the model and store tensors on.

  Returns:
      List[torch.Tensor]: A list of tensors, each of shape (n_layers * mlp_hidden_size),
                          containing the concatenated mean activations for all layers for each example.
  """
  all_activations = []
  for i, example in enumerate(examples):
    neuron_activations_example = []
    
    module_hooks = [(layer.act, get_extract_mean_activation_hook(neuron_activations_example)) for layer in model.transformer.blocks]

    with torch.no_grad(), add_forward_hooks(module_hooks):
      input_ids = torch.tensor(tokenizer.encode(example, out_type=int), device=device).unsqueeze(0)
      _ = model(input_ids)

    activations_all_layers = torch.cat(neuron_activations_example, dim=0)
    all_activations.append(activations_all_layers)
  
  return torch.stack(all_activations, dim=0)

def compute_average_precisions(activations_finnish: torch.Tensor, activations_english: torch.Tensor):
    all_activations = np.concatenate((activations_finnish.cpu().numpy(), activations_english.cpu().numpy()))
    labels_fi = np.concatenate((np.ones(activations_finnish.shape[0]), np.zeros(activations_english.shape[0])))
    labels_en = np.concatenate((np.zeros(activations_finnish.shape[0]), np.ones(activations_english.shape[0])))

    all_ap_scores_fi = []
    all_ap_scores_en = []

    num_neurons = activations_finnish.shape[1]
    for neuron_idx in range(num_neurons):
        neuron_activations = all_activations[:, neuron_idx]
        
        ap_score_fi = average_precision_score(labels_fi, neuron_activations)
        all_ap_scores_fi.append(ap_score_fi)

        ap_score_en = average_precision_score(labels_en, neuron_activations)
        all_ap_scores_en.append(ap_score_en)
    
    return np.array(all_ap_scores_fi), np.array(all_ap_scores_en)

def visualize_average_precisions(
    all_ap_scores_fi: np.ndarray, 
    all_ap_scores_en: np.ndarray, 
    num_neurons: int, 
    n_layers: int, 
    k: int, 
    output_filepath: str
):
    """
    Generates and saves a visualization of the distribution of top-k and middle-k 
    performing models (neurons) across different layers for Finnish and English data.

    The visualization consists of three histograms, arranged side-by-side:
    1.  Distribution of top-k neurons for Finnish data.
    2.  Distribution of middle-k neurons (centered around the median).
    3.  Distribution of top-k neurons for English data.

    Args:
        all_ap_scores_fi (np.ndarray): An array of average precision scores for each neuron for Finnish data.
        all_ap_scores_en (np.ndarray): An array of average precision scores for each neuron for English data.
        num_neurons (int): The total number of neurons.
        n_layers (int): The number of layers in the model.
        k (int): The number of top-performing neurons to consider.
        output_filepath (str): The file path where the generated visualization will be saved.
    
    Returns:
        counts_fi (np.ndarray): Histogram counts of top-k neurons for Finnish data across layers.
        counts_middle (np.ndarray): Histogram counts of middle-k neurons across layers.
        counts_en (np.ndarray): Histogram counts of top-k neurons for English data across layers.
        top_k_scores_fi (np.ndarray): Top-k average precision scores for Finnish data.
        top_k_scores_en (np.ndarray): Top-k average precision scores for English data.
    """
    top_k_indices_fi = np.argsort(all_ap_scores_fi)[-k:]
    top_k_scores_fi = all_ap_scores_fi[top_k_indices_fi]
    top_k_indices_en = np.argsort(all_ap_scores_en)[-k:]
    top_k_scores_en = all_ap_scores_en[top_k_indices_en]
    
    center_index = num_neurons // 2
    half_k = k // 2
    middle_k_indices = np.argsort(all_ap_scores_fi)[center_index - half_k : center_index + half_k]
    logging.info(f"Top-{k} and Middle-{k} indices calculated.")

    bin_size = num_neurons / n_layers
    bins = np.arange(0, num_neurons + 1, bin_size)
    
    counts_fi, _ = np.histogram(top_k_indices_fi, bins=bins)
    counts_en, _ = np.histogram(top_k_indices_en, bins=bins)
    counts_middle, _ = np.histogram(middle_k_indices, bins=bins)
    
    print(f"Count of top-k models in each layer (Finnish): {counts_fi}")
    print(f"Count of top-k models in each layer (English): {counts_en}")
    print(f"Count of middle-k models in each layer: {counts_middle}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        
    tick_positions = bins[:-1] + bin_size / 2
    tick_labels = [f'{i+1}' for i in range(n_layers)]

    axes[0].bar(bins[:-1], counts_fi, width=bin_size, align='edge', edgecolor='black', color='skyblue', alpha=0.8, label=f'Top-{k} Neurons')
    axes[0].set_title('Finnish', fontsize=18)
    axes[0].set_xlabel("Layer", fontsize=14)
    axes[0].set_ylabel(f"Count of Neurons", fontsize=14)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')

    axes[1].bar(bins[:-1], counts_middle, width=bin_size, align='edge', edgecolor='black', color='lightcoral', alpha=0.8, label=f'Middle-{k} Neurons')
    axes[1].set_title('Middle', fontsize=18)
    axes[1].set_xlabel("Layer", fontsize=14)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')

    axes[2].bar(bins[:-1], counts_en, width=bin_size, align='edge', edgecolor='black', color='lightgreen', alpha=0.8, label=f'Top-{k} Neurons')
    axes[2].set_title('English', fontsize=18)
    axes[2].set_xlabel("Layer", fontsize=14)
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels)
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')

    fig.suptitle(f"Distribution of Top-{k} and Middle-{k} Neurons Across Layers", fontsize=22, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_filepath, bbox_inches='tight', dpi=300)
    logging.info(f"Visualization saved to {output_filepath}")
    plt.close(fig)

    return counts_fi, counts_middle, counts_en, top_k_scores_fi[-10:], top_k_scores_en[-10:]

def visualize_ap_distribution(
    all_ap_scores_fi: np.ndarray, 
    all_ap_scores_en: np.ndarray, 
    output_filepath_prefix: str,
    checkpoint_name: str = "model" 
):
    """
    Generates and saves two plots showing the distribution of 
    Average Precision (AP) scores for Finnish and English data.

    Args:
        all_ap_scores_fi (np.ndarray): AP scores for Finnish data.
        all_ap_scores_en (np.ndarray): AP scores for English data.
        output_filepath_prefix (str): The prefix for the output file paths.
                                    Plots will be saved as 
                                    <prefix>_fi_ap_dist.png and <prefix>_en_ap_dist.png.
        checkpoint_name (str): Name of the checkpoint or model (e.g., "untrained", "step1000")
                            to be included in the plot title.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    ax_fi.hist(all_ap_scores_fi, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    ax_fi.set_title(f'Distribution of AP Scores for Finnish Data ({checkpoint_name})', fontsize=18)
    ax_fi.set_xlabel('Average Precision Score', fontsize=14)
    ax_fi.set_ylabel('Density', fontsize=14)
    ax_fi.tick_params(axis='both', which='major', labelsize=12)
    ax_fi.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    
    output_filepath_fi = f"{output_filepath_prefix}_fi_ap_dist.png"
    plt.tight_layout()
    plt.savefig(output_filepath_fi, dpi=300, bbox_inches='tight')
    logging.info(f"Finnish AP score distribution plot saved to {output_filepath_fi}")
    plt.close(fig_fi)

    fig_en, ax_en = plt.subplots(figsize=(10, 6))
    ax_en.hist(all_ap_scores_en, bins=100, density=True, color='lightcoral', edgecolor='black', alpha=0.7)
    ax_en.set_title(f'Distribution of AP Scores for English Data ({checkpoint_name})', fontsize=18)
    ax_en.set_xlabel('Average Precision Score', fontsize=14)
    ax_en.set_ylabel('Density', fontsize=14)
    ax_en.tick_params(axis='both', which='major', labelsize=12)
    ax_en.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')

    output_filepath_en = f"{output_filepath_prefix}_en_ap_dist.png"
    plt.tight_layout()
    plt.savefig(output_filepath_en, dpi=300, bbox_inches='tight')
    logging.info(f"English AP score distribution plot saved to {output_filepath_en}")
    plt.close(fig_en)

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

    # Hardcoded steps for checkpoints due to laziness.
    start_step = 1000
    end_step = 1000
    checkpoint_files = [args.checkpoints_dir + f"/step{i}" for i in range(start_step, end_step, 1000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    
    if not checkpoint_files:
        logging.error(f"No checkpoint files found based on the provided directory and steps. Please check the path and parameters.")
        return
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint files to process.")

    finnish_sentences = pd.read_parquet(args.finnish_data_path)["text"].to_list()
    english_sentences = pd.read_parquet(args.english_data_path)["text"].to_list()

    all_counts = []

    config = ModelConfig()
    model_untrained = OLMo(config)
    model_untrained.to(device)

    try:
        num_layers = len(model_untrained.transformer.blocks)

        activations_finnish = get_activations(
            model_untrained, tokenizer, finnish_sentences, device)
        activations_english = get_activations(
            model_untrained, tokenizer, english_sentences, device)
        logging.info(f"Activations for untrained model calculated successfully.")
        
        all_ap_scores_fi, all_ap_scores_en = compute_average_precisions(activations_finnish, activations_english)
        logging.info(f"Average precision scores calculated for untrained model.")

        base_figure_name_untrained = f"viz_neurons_{args.experiment_name}_untrained"
        base_figure_save_path_untrained = os.path.join(args.output_dir, base_figure_name_untrained)
        logging.info(f"Visualizing average precisions for untrained model...")
        counts_fi, counts_middle, counts_en, top_scores_fi, top_scores_en = visualize_average_precisions(
            all_ap_scores_fi, all_ap_scores_en,
            num_neurons=all_ap_scores_fi.shape[0],
            n_layers=num_layers,
            k=1000,
            output_filepath=f"{base_figure_save_path_untrained}.png"
        )

        visualize_ap_distribution(all_ap_scores_fi, all_ap_scores_en, os.path.join(args.output_dir, "untrained"), "untrained")

        with open(os.path.join(args.output_dir, f"neurons_experiment_{args.experiment_name}_untrained.json"), 'w') as f:
            json.dump({
                "checkpoint": "untrained",
                "counts_fi": counts_fi.tolist(),
                "counts_middle": counts_middle.tolist(),
                "counts_en": counts_en.tolist(),
                "top_scores_fi": top_scores_fi.tolist(),
                "top_scores_en": top_scores_en.tolist(),
                "all_ap_scores_fi": all_ap_scores_fi.tolist(),
                "all_ap_scores_en": all_ap_scores_en.tolist(),
                "num_fi_50": str((all_ap_scores_fi >= 0.50).sum()),
                "num_fi_75": str((all_ap_scores_fi >= 0.75).sum()),
                "num_fi_90": str((all_ap_scores_fi >= 0.90).sum()),
                "num_fi_95": str((all_ap_scores_fi >= 0.95).sum()),
                "num_fi_99": str((all_ap_scores_fi >= 0.99).sum()),
                "num_en_50": str((all_ap_scores_en >= 0.50).sum()),
                "num_en_75": str((all_ap_scores_en >= 0.75).sum()),
                "num_en_90": str((all_ap_scores_en >= 0.90).sum()),
                "num_en_95": str((all_ap_scores_en >= 0.95).sum()),
                "num_en_99": str((all_ap_scores_en >= 0.99).sum()),
            }, f, indent=4)

        all_counts.append({
            "checkpoint": "untrained",
            "counts_fi": counts_fi.tolist(),
            "counts_middle": counts_middle.tolist(),
            "counts_en": counts_en.tolist(),
            "top_scores_fi": top_scores_fi.tolist(),
            "top_scores_en": top_scores_en.tolist(),
            "num_fi_90": str((all_ap_scores_fi >= 0.90).sum()),
            "num_fi_95": str((all_ap_scores_fi >= 0.95).sum()),
            "num_fi_99": str((all_ap_scores_fi >= 0.99).sum()),
            "num_en_90": str((all_ap_scores_en >= 0.90).sum()),
            "num_en_95": str((all_ap_scores_en >= 0.95).sum()),
            "num_en_99": str((all_ap_scores_en >= 0.99).sum()),
        })

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
                model_checkpoint, tokenizer, finnish_sentences, device)
            activations_english = get_activations(
                model_checkpoint, tokenizer, english_sentences, device)
            logging.info(f"Activations for checkpoint {checkpoint_path} calculated successfully.")
            
            all_ap_scores_fi, all_ap_scores_en = compute_average_precisions(activations_finnish, activations_english)
            logging.info(f"Average precision scores calculated for checkpoint {checkpoint_path}.")

            checkpoint_name_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
            base_figure_name = f"viz_neurons_{args.experiment_name}_step{checkpoint_name_stem}"
            base_figure_save_path = os.path.join(args.output_dir, base_figure_name)
            logging.info(f"Visualizing average precisions for checkpoint {checkpoint_path}...")
            counts_fi, counts_middle, counts_en, top_scores_fi, top_scores_en = visualize_average_precisions(
                all_ap_scores_fi, all_ap_scores_en,
                num_neurons=all_ap_scores_fi.shape[0],
                n_layers=num_layers,
                k=1000,
                output_filepath=f"{base_figure_save_path}.png"
            )

            visualize_ap_distribution(all_ap_scores_fi, all_ap_scores_en, os.path.join(args.output_dir, checkpoint_name_stem), checkpoint_name_stem)

            all_counts.append({
                "checkpoint": checkpoint_path,
                "counts_fi": counts_fi.tolist(),
                "counts_middle": counts_middle.tolist(),
                "counts_en": counts_en.tolist(),
                "top_scores_fi": top_scores_fi.tolist(),
                "top_scores_en": top_scores_en.tolist(),
                "num_fi_90": str((all_ap_scores_fi >= 0.90).sum()),
                "num_fi_95": str((all_ap_scores_fi >= 0.95).sum()),
                "num_fi_99": str((all_ap_scores_fi >= 0.99).sum()),
                "num_en_90": str((all_ap_scores_en >= 0.90).sum()),
                "num_en_95": str((all_ap_scores_en >= 0.95).sum()),
                "num_en_99": str((all_ap_scores_en >= 0.99).sum()),
            })

            with open(os.path.join(args.output_dir, f"neurons_experiment_{args.experiment_name}_{checkpoint_name_stem}.json"), 'w') as f:
                json.dump({
                    "checkpoint": checkpoint_path,
                    "counts_fi": counts_fi.tolist(),
                    "counts_middle": counts_middle.tolist(),
                    "counts_en": counts_en.tolist(),
                    "top_scores_fi": top_scores_fi.tolist(),
                    "top_scores_en": top_scores_en.tolist(),
                    "all_ap_scores_fi": all_ap_scores_fi.tolist(),
                    "all_ap_scores_en": all_ap_scores_en.tolist(),
                    "num_fi_50": str((all_ap_scores_fi >= 0.50).sum()),
                    "num_fi_75": str((all_ap_scores_fi >= 0.75).sum()),
                    "num_fi_90": str((all_ap_scores_fi >= 0.90).sum()),
                    "num_fi_95": str((all_ap_scores_fi >= 0.95).sum()),
                    "num_fi_99": str((all_ap_scores_fi >= 0.99).sum()),
                    "num_en_50": str((all_ap_scores_en >= 0.50).sum()),
                    "num_en_75": str((all_ap_scores_en >= 0.75).sum()),
                    "num_en_90": str((all_ap_scores_en >= 0.90).sum()),
                    "num_en_95": str((all_ap_scores_en >= 0.95).sum()),
                    "num_en_99": str((all_ap_scores_en >= 0.99).sum()),
                }, f, indent=4)

            print(all_counts[-1])
        
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
    output_json_path = os.path.join(args.output_dir, f"neurons_experiment_{args.experiment_name}.json")
    with open(output_json_path, 'w') as f:
        json.dump(all_counts, f, indent=4)
    logging.info(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model checkpoints to visualize and save language-specific activations.")
    
    parser.add_argument("--finnish_data_path", type=str, required=True, help="Path to the Parquet file containing Finnish examples.")
    parser.add_argument("--english_data_path", type=str, required=True, help="Path to the Parquet file containing English examples.")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model file.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing model checkpoint files.")
    parser.add_argument("--last_step", type=int, default=47000, help="Last step number for checkpoints to process. Default is 47000.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output figures.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment for output files.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Autodetects CUDA if None.")
    args = parser.parse_args()
        
    main(args)
