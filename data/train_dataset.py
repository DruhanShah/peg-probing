import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

from .PEG import PEG


class TrainDataset(Dataset, ABC):
    """Base class for training datasets on formal languages."""
    
    def __init__(self,
                 language,
                 num_samples,
                 max_len, 
                 seed = 42,
                 **kwargs):
        self.language = language
        self.num_samples = num_samples
        self.max_len = max_len
        self.seed = seed
        
        self.PEG = PEG(language, max_length=max_len)
        self.pad_token = "<pad>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]
        
        self.data = []
        self.labels = []
        self._generated = False
    
    @abstractmethod
    def generate_data(self):
        pass
    
    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        dataset_type = "recognizer" if hasattr(self, '_is_binary') else "generator"
        data_path = os.path.join(base_dir, dataset_type, self.language, "dataset.pkl")
        
        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data["data"]
            self.labels = saved_data["labels"]
            self._generated = True
    
    def length_stats(self):
        if hasattr(self, '_is_binary'):
            lengths = {
                "pos": [0] * (self.max_len + 1),
                "neg": [0] * (self.max_len + 1)
            }
            for string, label in zip(self.data, self.labels):
                key = "pos" if label == 1 else "neg"
                lengths[key][len(string)] += 1
            return lengths
        else:
            lengths = [0] * (self.max_len + 1)
            for string in self.data:
                lengths[len(string)] += 1
            return lengths
    
    def __len__(self):
        return len(self.data) if self._generated else self.num_samples


class RecognizerDataset(TrainDataset):
    """Dataset for training binary classifiers on formal languages."""
    
    def __init__(self,
                 language,
                 num_samples,
                 max_len, 
                 seed = 42,
                 pos_ratio = 0.5,
                 **kwargs):
        super().__init__(language, num_samples, max_len, seed, **kwargs)
        self.pos_ratio = pos_ratio
        self._is_binary = True
    
    def generate_data(self, quiet=False):
        if self._generated:
            return
            
        num_positive = int(self.num_samples * self.pos_ratio)
        num_negative = self.num_samples - num_positive
        
        # Generate positive samples
        for _ in tqdm(range(num_positive),
                      desc="Generating positive samples",
                      disable=quiet):
            target_length = random.choices(
                self.PEG.valid_lengths,
                weights = self.PEG.length_weights,
            )[0]
            sequence = self.PEG.positive_generator(target_length)
            self.data.append(sequence)
            self.labels.append(1)
        
        # Generate negative samples
        for _ in tqdm(range(num_negative),
                      desc="Generating negative samples",
                      disable=quiet):
            target_length = random.randint(2, self.max_len)
            sequence = self.PEG.negative_generator(target_length)
            self.data.append(sequence)
            self.labels.append(0)
        
        # Shuffle the combined data
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)
        
        self._generated = True
    
    def __getitem__(self, index):
        if not self._generated:
            raise RuntimeError("Data not generated. Call generate_data() first.")
            
        sequence = self.data[index]
        label = self.labels[index]
        
        sequence_tokens = torch.tensor(
            self.PEG.tokenize_string(sequence),
            dtype=torch.long
        )
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return sequence_tokens, label_tensor

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        max_len = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = torch.cat([
                seq,
                torch.full((max_len - len(seq),), self.pad_token_id, dtype=torch.long)
            ])
            padded_sequences.append(padded_seq)
        
        return {
            'inputs': torch.stack(padded_sequences),
            'outputs': torch.stack(list(labels)),
        }


class GeneratorDataset(TrainDataset):
    """Dataset for training autoregressive language models on formal languages."""
    
    def __init__(self,
                 language,
                 num_samples,
                 max_len, 
                 seed = 42,
                 pos_ratio = 1.0,
                 **kwargs):
        super().__init__(language, num_samples, max_len, seed, **kwargs)
        self.pos_ratio = pos_ratio
    
    def generate_data(self, quiet=False):
        if self._generated:
            return
            
        if self.pos_ratio == 1.0:
            for _ in tqdm(range(self.num_samples),
                          desc="Generating positive samples",
                          disable=quiet):
                target_length = random.choices(
                    self.PEG.valid_lengths,
                    weights = self.PEG.length_weights,
                )[0]
                sequence = self.PEG.positive_generator(target_length)
                self.data.append(sequence)
                self.labels.append(1)
        else:
            num_positive = int(self.num_samples * self.pos_ratio)
            num_negative = self.num_samples - num_positive
            
            # Generate positive samples
            for _ in tqdm(range(num_positive),
                          desc="Generating positive samples",
                          disable=quiet):
                target_length = random.choices(
                    self.PEG.valid_lengths,
                    weights = self.PEG.length_weights,
                )[0]
                sequence = self.PEG.positive_generator(target_length)
                self.data.append(sequence)
                self.labels.append(1)
            
            # Generate negative samples
            for _ in tqdm(range(num_negative),
                          desc="Generating negative samples",
                          disable=quiet):
                target_length = random.randint(2, self.max_len)
                sequence = self.PEG.negative_generator(target_length)
                self.data.append(sequence)
                self.labels.append(0)
            
            # Shuffle the data
            combined = list(zip(self.data, self.labels))
            random.shuffle(combined)
            self.data, self.labels = zip(*combined)
            self.data = list(self.data)
            self.labels = list(self.labels)
        
        self._generated = True
    
    def __getitem__(self, index):
        if not self._generated:
            raise RuntimeError("Data not generated. Call generate_data() first.")
            
        sequence = self.data[index]
        
        # Tokenize the sequence (includes <eos> token)
        tokens = self.PEG.tokenize_string(sequence)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        if len(tokens_tensor) > 1:
            input_tokens = tokens_tensor[:-1]
            target_tokens = tokens_tensor[1:]
        else:
            input_tokens = torch.tensor([self.pad_token_id], dtype=torch.long)
            target_tokens = tokens_tensor
        
        return input_tokens, target_tokens
    
    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        
        max_input_len = max(len(seq) for seq in inputs)
        max_target_len = max(len(seq) for seq in targets)
        max_len = max(max_input_len, max_target_len)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            inp_padded = torch.cat([
                inp, 
                torch.full((max_len - len(inp),), self.pad_token_id, dtype=torch.long)
            ])
            
            tgt_padded = torch.cat([
                tgt,
                torch.full((max_len - len(tgt),), self.pad_token_id, dtype=torch.long)
            ])
            
            padded_inputs.append(inp_padded)
            padded_targets.append(tgt_padded)
        
        return {
            'inputs': torch.stack(padded_inputs),
            'outputs': torch.stack(padded_targets),
        }


def create_train_dataset(type, **kwargs):
    if type.lower() == "recognizer":
        return RecognizerDataset(**kwargs)
    elif type.lower() == "generator":
        return GeneratorDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {type}. Must be 'recognizer' or 'generator'.")
