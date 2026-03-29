import argparse
import torch
from ....training.olmo.model import OLMo
from ....training.olmo.tokenizer import Tokenizer
from .hooks import get_add_control_vector_all_hook, add_forward_hooks
from typing import List, Optional, Dict
import warnings
import os
import csv
import json
import random

warnings.filterwarnings("ignore", category=UserWarning)

random.seed(42)

def load_rocstories_data(csv_path: str) -> List[dict]:
    """
    Reads the rocstories.csv file and prepares the prompts.

    Args:
        csv_path (str): The path to the rocstories.csv file.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary has a 'ctx' key
                    containing the processed prompt.
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}")
        return examples

    print(f"Loading and processing prompts from {csv_path}...")
    try:
        with open(csv_path, "r", encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and len(row) >= 7:
                    first_four_sentences = row[2:6]
                    prompt = " ".join(first_four_sentences)
                    examples.append({'ctx': prompt})
    except Exception as e:
        print(f"Error reading or processing {csv_path}: {e}")

    if not examples:
        print("No examples were loaded.")
        return []
        
    print(f"Successfully loaded {len(examples)} prompts.")
    return examples

def generate_rocstories_completions(
    model: OLMo,
    olmo_tokenizer: Tokenizer,
    dataset: List[dict],
    generation_config: Dict,
    control_vector: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    a: float = 1.0
) -> List[dict]:
    """
    Generates controlled or uncontrolled completions for the provided dataset.
    """
    device = model.device

    if control_vector is not None:
        print(f"\n--- Generating Controlled Completions (a={a}) ---")
        active_layers = layers
        if active_layers is None:
            active_layers = list(range(model.config.n_layers))
        
        model_dim = model.config.d_model            
        module_hooks = [(layer, get_add_control_vector_all_hook(-a * control_vector[:, i*model_dim:(i+1)*model_dim])) for i, layer in enumerate(model.transformer.blocks) if i in active_layers]
    else:
        print("\n--- Generating Standard (No Control) Completions ---")
        module_hooks = []

    results = []
    for i, sample in enumerate(dataset):
        context = sample['ctx']
        tokens = olmo_tokenizer.encode(context, out_type=int, add_bos=True)
        if not tokens:
            print(f"Warning: Tokenizer returned empty tokens for sample {i+1}.")
            continue
            
        input_ids = torch.tensor(tokens, device=device)
        
        with add_forward_hooks(module_hooks):
            generated_tokens = model.generate(input_ids, **generation_config)
                
        generated_text = olmo_tokenizer.decode(generated_tokens.token_ids.tolist())
        generated_text = generated_text[len(context):]

        results.append({
            'context': context,
            'completion': generated_text
        })

    print(f"--- Completed Generation. Generated {len(results)} completions. ---")
    return results

def main(args):
    """
    Main function to orchestrate hyperparameter search for 'a' coefficient.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"All output files will be saved to: {args.output_dir}")

    try:
        olmo_tokenizer = Tokenizer(args.tokenizer_path)
    except Exception as e:
        print(f"Fatal Error: Could not load tokenizer. {e}")
        return
    
    #stop_sequences = [[olmo_tokenizer.piece_to_id(el)] for el in [".", "!", "?", r"\n"]]
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_steps": 30,
        "ignore_eos": True,
    }

    rocstories_data_full = load_rocstories_data(args.rocstories_path)
    if not rocstories_data_full:
        print("Fatal Error: No data loaded from ROCStories file. Exiting.")
        return

    if args.num_samples < len(rocstories_data_full):
        print(f"Randomly sampling {args.num_samples} examples from the dataset of {len(rocstories_data_full)}.")
        rocstories_data = random.sample(rocstories_data_full, args.num_samples)
    else:
        print(f"Using all {len(rocstories_data_full)} available examples.")
        rocstories_data = rocstories_data_full

    checkpoint_paths = [f"{args.checkpoints_dir}/step{step}" for step in range(40000, 50000, 10000)] + [f"{args.checkpoints_dir}/step{args.last_step}"]
    print(f"Found {len(checkpoint_paths)} checkpoints to process.")

    for checkpoint_path in checkpoint_paths:
        if not os.path.isdir(checkpoint_path):
            print(f"Skipping non-directory file: {checkpoint_path}")
            continue

        print(f"\n{'='*25} Processing checkpoint: {checkpoint_path} {'='*25}")
        checkpoint_basename = os.path.basename(checkpoint_path)

        try:
            model = OLMo.from_checkpoint(checkpoint_path, device=device)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: Could not load model from {checkpoint_path}. Skipping. Error: {e}")
            continue

        controlled_results_by_a = {}
        if args.control_vectors_dir and args.control_a_coefficients:
            control_vector_name = f"control_vector_mean_fineng_{checkpoint_basename}.pt"
            control_vector_path = os.path.join(args.control_vectors_dir, control_vector_name)

            if os.path.exists(control_vector_path):
                print(f"Found corresponding control vector: {control_vector_path}")
                control_vector = torch.load(control_vector_path).to(device)
                for a_coeff in args.control_a_coefficients:
                    controlled_results = generate_rocstories_completions(
                        model=model,
                        olmo_tokenizer=olmo_tokenizer,
                        dataset=rocstories_data,
                        generation_config=generation_config,
                        control_vector=control_vector,
                        layers=args.control_layers,
                        a=a_coeff
                    )
                    controlled_results_by_a[str(a_coeff)] = controlled_results
            else:
                raise FileNotFoundError(f"Warning: No corresponding control vector found at {control_vector_path}. Skipping controlled generation.")
        else:
            raise ValueError("No control vector directory or no 'a' coefficients provided.")
        
        results = {
            "checkpoint_path": checkpoint_basename,
            "controlled_results_by_a": controlled_results_by_a,
        }

        results_path = os.path.join(args.output_dir, f"{checkpoint_basename}/rocstories_results_{checkpoint_basename}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved combined hyperparameter search results to {results_path}")

    print(f"\n{'='*25} All checkpoints processed. {'='*25}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter search for 'a' coefficient in controlled text generation.")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--last_step", type=int, required=True, help="The last step number of the model to process.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples from the dataset to use for generation. Defaults to 200.")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--rocstories_path", type=str, required=True)
    parser.add_argument("--control_vectors_dir", type=str, default=None, help="Directory containing control vectors. Format: 'control_vector_mean_fineng_{checkpoint_basename}.pt'.")
    parser.add_argument("--control_layers", type=int, nargs="*", default=None, help="Layer indices for applying the control vector. Defaults to all layers.")
    parser.add_argument("--control_a_coefficients", nargs="*", type=float, default=None, help="A list of scaling factors 'a' for the control vector to test.")

    args = parser.parse_args()
    main(args)
