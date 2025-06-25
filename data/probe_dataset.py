import os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pickle as pkl
import random

from .PEG import PEG


class PSDataset():
    
    def __init__(self, language, precomp, num_iters,
                 max_len, seed, **other_args):

        self.num_iters = num_iters
        self.max_len = max_len
        self.seed = seed

        self.PEG = PEG(language, max_length=self.max_len)

        self.pad_token = "<eos>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.generated = precomp
        self.data = []
        self.labels = []

    def save_data(self, path_to_results, num_samples):

        for _ in tqdm(range(num_samples), desc="Generating positive samples"):
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)
            label = self.PEG.parse_state_generator(sequence)

            self.data.append(sequence)
            self.labels.append(label)
        
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)
        
        base_dir = os.path.join(path_to_results, "data")
        os.makedirs(base_dir, exist_ok=True)
        
        data_path = os.path.join(base_dir, f"{self.PEG.language}_parse.pkl")
        with open(data_path, "wb") as f:
            pkl.dump({
                "data": self.data,
                "labels": self.labels
            }, f)

        return len(self.data)

    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        data_path = os.path.join(base_dir, f"{self.PEG.language}_binary.pkl")
        
        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data["data"]
            self.labels = saved_data["labels"]

    def __len__(self):
        return len(self.data) if self.generated else self.num_iters

    def __getitem__(self, index):
        if self.generated:
            sequence = self.data[index]
            label = self.labels[index]
        else:
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)
            label = self.PEG.parse_state_generator(sequence)
        
        sequence_tokens = torch.tensor(
            self.PEG.tokenize_string(sequence),
            dtype=torch.long
        )
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return sequence_tokens, label_tensor
