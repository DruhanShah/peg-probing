import torch
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import os
from .PEG import PEG
import pickle as pkl


class PEGDataset():
    
    def __init__(
        self, language, config, precomp,
        num_iters, max_len, seed,
    ):

        self.num_iters = num_iters
        self.max_len = max_len

        self.PEG = PEG(language)

        self.pad_token = "<pad>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.generated = precomp
        self.generator = self.PEG.string_generator(num_samples=self.num_iters)

        self.template = torch.tensor(self.PEG.tokenize_string("")[:-1])

    def save_grammar(self, path_to_results):
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PEG.pkl"), "wb") as f:
            pkl.dump(self.PEG, f)

    def load_grammar(self, path_to_results):
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PEG.pkl"), "rb") as f:
            self.PEG = pkl.load(f)

    def save_data(self, path_to_results, num_samples):
        self.data = []
        prefix_freqs = {i: 0 for i in range(1, self.max_len+1)}
        for _ in tqdm(range(num_samples)):
            sequence, pref = self.generator.__next__()
            self.data.append(sequence)
            prefix_freqs[pref] += 1
        
        base_dir = os.path.join(path_to_results, "data")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, f"{self.PEG.language}.pkl"), "wb") as f:
            print(f"Saving data to {base_dir}/{self.PEG.language}.pkl")
            pkl.dump(self.data, f)

        return prefix_freqs

    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        with open(os.path.join(base_dir, f"{self.PEG.language}.pkl"), "rb") as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data) if self.generated else self.num_iters

    def __getitem__(self, index):        
        if self.generated:
            sequence = self.data[index]
        else:
            sequence, _ = self.generator.__next__()
        sequence = torch.tensor(self.PEG.tokenize_string(sequence))
        seq_length = float(sequence.size(0))
        return sequence, seq_length


def get_dataloader(
        language, config, num_iters, max_len,
        precomp, results_dir,
        seed = 42,
        batch_size = 32,
        num_workers = 0,
):
    dataset = PEGDataset(
        language=language,
        config=config,
        precomp=precomp,
        num_iters=num_iters,
        max_len=max_len,
        seed=seed,
    )
    dataset.load_data(results_dir)

    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader

