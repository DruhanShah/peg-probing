import os
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pickle as pkl
import random
from abc import ABC, abstractmethod

from .PEG import PEG


class ProbeDataset(ABC, Dataset):

    def __init__(self,
                 language,
                 precomp,
                 num_iters,
                 max_len,
                 seed,
                 **kwargs):
        self.language = language
        self.num_iters = num_iters
        self.max_len = max_len
        self.seed = seed

        self.PEG = PEG(language, max_length=self.max_len)
        self.pad_token = "<pad>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.data = []
        self.labels = []
        self._generated = False

    def generate_data(self, quiet=False):
        if self._generated:
            return

        for _ in tqdm(range(self.num_iters),
                      desc="Generating positive samples",
                      disable=quiet):
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)
            label = self._generate_label(sequence)
            label = self.PEG.parse_state_generator(sequence)

            self.data.append(sequence)
            self.labels.append(label)

        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)

        self._generated = True

    @abstractmethod
    def _generate_label(self, sequence):
        """Generate labels for the dataset."""
        pass

    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        data_path = os.path.join(base_dir, f"{self.PEG.language}_binary.pkl")
        
        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data["data"]
            self.labels = saved_data["labels"]
            self._generated = True

    def __len__(self):
        return len(self.data) if self._generated else self.num_iters

    def __getitem__(self, index):
        if self._generated:
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

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        
        max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        padded_labels = []

        for inp, tgt in zip(sequences, labels):
            pad_len_inp = max_length - len(inp)
            pad_len_tgt = max_length - len(tgt)
            padded_seq = torch.cat(
                [inp, torch.tensor([self.pad_token_id] * pad_len_inp)]
            )
            padded_label = torch.cat(
                [tgt, torch.tensor([-1] * pad_len_tgt)]
            )
            
            padded_sequences.append(padded_seq)
            padded_labels.append(padded_label)

        return {
            "inputs": torch.stack(padded_sequences),
            "outputs": torch.stack(padded_labels)
        }


class ParseStateDataset(ProbeDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_label(self, sequence):
        return self.PEG.parse_state_generator(sequence)


class ParseDepthDataset(ProbeDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _generate_label(self, sequence):
        return self.PEG.parse_depth_generator(sequence)


class TokenCategoryDataset(ProbeDataset):

    def __init__(self, category=None, **kwargs):
        super().__init__(**kwargs)
        self.category = category

    def _generate_label(self, sequence):
        return self.PEG.token_category_generator(sequence, self.category)


def create_probe_dataset(type, **kwargs):
    if type.lower() == "parse_state":
        return ParseStateDataset(**kwargs)
    if type.lower() == "parse_depth":
        return ParseDepthDataset(**kwargs)
    if type.lower() == "token_category":
        return TokenCategoryDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {type}")
