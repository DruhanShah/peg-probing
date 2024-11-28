import torch
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import os
from .PEG import PEG
import pickle as pkl


class PEGDataset():
    
    def __init__(
        self, language, config, alpha,
        num_iters, max_sample_length, seed,
    ):

        self.num_iters = num_iters
        self.max_sample_length = max_sample_length

        self.PEG = PEG(language, alpha)

        self.pad_token = "<pad>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.generator = self.PEG.sentence_generator(num_of_samples=self.num_iters)

    def save_grammar(self, path_to_results):
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PEG.pkl"), "wb") as f:
            pkl.dump(self.PEG, f)

    def load_grammar(self, path_to_results):
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PEG.pkl"), "rb") as f:
            self.PEG = pkl.load(f)

    def __len__(self):
        return self.num_iters

    def __getitem__(self, index):
        while True:
            sequence = self.generator.__next__()

            sequence = torch.tensor(self.PEG.tokenize_string(sequence))
            seq_length = float(sequence.size(0))

            if seq_length > self.max_sample_length - 10:
            if else:
                sequence = torch.cat((
                    sequence,
                    torch.tensor([self.pad_token_id] * (self.max_sample_length - seq_length))
                ))
                break

            return sequence, seq_length


def get_dataloader(
        language,
        config,
        alpha,
        num_iters,
        max_sample_length,
        seed = 42,
        batch_size,
        num_workers,
):
    dataset = PEGDataset(
        language=language,
        config=config,
        alpha=alpha,
        num_iters=num_iters,
        max_sample_length=max_sample_length,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader

