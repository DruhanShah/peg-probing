import torch
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import os
from .PEG import PEG
import pickle as pkl
import random

from utils import obj_to_dict


class PEGDataset():
    
    def __init__(self, language, precomp, num_iters, max_len, seed):

        self.num_iters = num_iters
        self.max_len = max_len
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.PEG = PEG(language, max_length=self.max_len)

        self.pad_token = "<bos>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.generated = precomp

    def save_data(self, path_to_results, num_samples):
        self.data = []
        self.labels = []
        
        pos_samples = num_samples // 2
        neg_samples = num_samples - pos_samples
        
        for _ in tqdm(range(pos_samples), desc="Generating positive samples"):
            target_length = random.randint(1, self.max_len)
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
                'data': self.data,
                'labels': self.labels
            }, f)

        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        length_stats = {
            'positive_samples': pos_count,
            'negative_samples': neg_count,
            'total_samples': len(self.labels)
        }
        
        return length_stats

    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        data_path = os.path.join(base_dir, f"{self.PEG.language}_binary.pkl")
        
        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data['data']
            self.labels = saved_data['labels']

    def __len__(self):
        return len(self.data) if self.generated else self.num_iters

    def __getitem__(self, index):
        if self.generated:
            sequence = self.data[index]
            label = self.labels[index]
        else:
            if index % 2 == 0:
                # Generate positive sample
                sequence, _ = next(self.pos_generator)
                label = 1
            else:
                # Generate negative sample
                target_length = random.randint(0, self.max_len)
                sequence = self.generate_negative_sample(target_length)
                label = 0
        
        sequence_tokens = torch.tensor(self.PEG.tokenize_string(sequence))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return sequence_tokens, label_tensor


def get_dataloader(cfg, seed=42):
    dataset = PEGDataset(**obj_to_dict(cfg), seed=seed)
    if cfg.precomp:
        dataset.load_data()

    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    return dataloader


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    sequences, labels = zip(*batch)
    
    max_len = max(seq.size(0) for seq in sequences)
    
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        pad_length = max_len - seq.size(0)
        if pad_length > 0:
            # Assuming pad_token_id is 0 (index of <bos>)
            padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=seq.dtype)])
        else:
            padded_seq = seq
            
        attention_mask = torch.cat([
            torch.ones(seq.size(0), dtype=torch.long),
            torch.zeros(pad_length, dtype=torch.long)
        ])
        
        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)
    
    sequences_tensor = torch.stack(padded_sequences)
    attention_masks_tensor = torch.stack(attention_masks)
    labels_tensor = torch.stack(list(labels))
    
    return {
        'input_ids': sequences_tensor,
        'attention_mask': attention_masks_tensor,
        'labels': labels_tensor
    }
