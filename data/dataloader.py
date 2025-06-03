import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import os
from .PEG import PEG
import pickle as pkl
import random


class PEGDataset():
    
    def __init__(self, language, precomp, num_iters,
                 max_len, seed, **other_args):

        self.num_iters = num_iters
        self.max_len = max_len
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.PEG = PEG(language, max_length=self.max_len)

        self.pad_token = "<eos>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.generated = precomp

    def save_data(self, path_to_results, num_samples):
        self.data = []
        self.labels = []
        
        pos_samples = num_samples // 2
        neg_samples = num_samples - pos_samples
        
        for _ in tqdm(range(pos_samples), desc="Generating positive samples"):
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)

            self.data.append(sequence)
            self.labels.append(1)
        
        for _ in tqdm(range(neg_samples), desc="Generating negative samples"):
            target_length = random.randint(1, self.max_len)
            sequence = self.PEG.negative_generator(target_length)
            
            self.data.append(sequence)
            self.labels.append(0)
        
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)
        
        base_dir = os.path.join(path_to_results, "data")
        os.makedirs(base_dir, exist_ok=True)
        
        data_path = os.path.join(base_dir, f"{self.PEG.language}_binary.pkl")
        with open(data_path, "wb") as f:
            print(f"Saving data to {data_path}")
            pkl.dump({
                "data": self.data,
                "labels": self.labels
            }, f)

        lengths = {
            "pos": [0 for i in range(self.max_len+1)],
            "neg": [0 for i in range(self.max_len+1)],
        }
        for string, label in zip(self.data, self.labels):
            key = "pos" if label == 1 else "neg"
            lengths[key][len(string)] += 1

        return lengths

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
            if index % 2 == 0:
                # Generate positive sample
                target_length = random.choice(self.PEG.valid_lengths)
                sequence = self.PEG.positive_generator(target_length)
                label = 1
            else:
                # Generate negative sample
                target_length = random.randint(1, self.max_len)
                sequence = self.PEG.negative_generator(target_length)
                label = 0
        
        sequence_tokens = torch.tensor(self.PEG.tokenize_string(sequence))
        label_tensor = torch.tensor(label, dtype=torch.float)
        
        return sequence_tokens, label_tensor


def get_dataloader(cfg, work_dir, seed=42):
    dataset = PEGDataset(**OmegaConf.to_object(cfg), seed=seed)
    if cfg.precomp:
        dataset.load_data(work_dir)

    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    sequences, labels = zip(*batch)
    
    max_len = max(seq.size(0) for seq in sequences)
    
    padded_sequences = []
    
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            padded_seq = torch.cat([seq, torch.zeros(pad_len, dtype=seq.dtype)])
        else:
            padded_seq = seq
            
        padded_sequences.append(padded_seq)
    
    sequences_tensor = torch.stack(padded_sequences)
    labels_tensor = torch.stack(list(labels))
    
    return {
        'input_ids': sequences_tensor,
        'labels': labels_tensor
    }
