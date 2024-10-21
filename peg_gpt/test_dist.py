import argparse, sys
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig
)
from data import LangDataset
import torch
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate(lang, dataset, model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", type=str)
    parser.add_argument("work_dir", type=str)
    args = parser.parse_args()
    
    dataset = get_data(args.lang, args.work_dir)
    model = HookedTransformer(HookedTransformerConfig.load(f"{args.work_dir}/models/{args.lang}.pt"))

    generate(args.lang, dataset, model)
