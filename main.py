import argparse
import yaml
from pathlib import Path

from src.data.generate_cohort import generate_cohort_mimic
from src.inference.generate_predictions import generate_predictions, generate_baseline_predictions

ALL_MODELS = ["google/gemma-3-4b-it", "google/gemma-3-27b-it", "google/medgemma-27b-text-it",
              "openai/gpt-oss-20b", "openai/gpt-oss-120b",
              "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
              "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
              "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ehrLLM experiments")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default="./configs", help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['generate_cohort', 'test', 'baseline', 'iteration'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(Path(args.config) / f"{args.experiment}.yaml") if args.config else {}

    if args.mode == 'generate_cohort':
        generate_cohort_mimic(config) # Filter MEDS input to generate downstream evaluation cohort
    elif args.mode == 'baseline':
        generate_baseline_predictions(config)
    elif args.mode == 'iteration':
        for model in ALL_MODELS:
            for iteration in range(10):
                config['model_id'] = model
                config['explicit_missingness'] = False
                config['include_cot_prompt'] = False
                config['iteration'] = iteration + 1
                try:
                    generate_predictions(config)
                except Exception as e:
                    print("Error with", model)
                    print(e)
    else:
        for model in ALL_MODELS:
            config['model_id'] = model
            config['iteration'] = 0
            for explicit_missingness in [True, False]:
                config['explicit_missingness'] = explicit_missingness
                for include_cot_prompt in [True, False]:
                    config['include_cot_prompt'] = include_cot_prompt
                    try:
                        generate_predictions(config)
                    except Exception as e:
                        print("Error with", model)
                        print(e)