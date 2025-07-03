import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import random
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
        self.pad_token = "<eos>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]
        
        self.data = []
        self.labels = []
        self._generated = False
    
    @abstractmethod
    def generate_data(self):
        pass
    
    def save_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        os.makedirs(base_dir, exist_ok=True)
        
        # Determine filename based on dataset type
        dataset_type = "binary" if hasattr(self, '_is_binary') else "string"
        data_path = os.path.join(base_dir, f"{self.language}_{dataset_type}.pkl")
        
        save_dict = {
            "data": self.data,
            "labels": self.labels,
            "language": self.language,
            "max_len": self.max_len,
            "seed": self.seed
        }
        
        with open(data_path, "wb") as f:
            pkl.dump(save_dict, f)
        
        return self._compute_length_stats()
    
    def load_data(self, path_to_results):
        base_dir = os.path.join(path_to_results, "data")
        dataset_type = "binary" if hasattr(self, '_is_binary') else "string"
        data_path = os.path.join(base_dir, f"{self.language}_{dataset_type}.pkl")
        
        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data["data"]
            self.labels = saved_data["labels"]
            self._generated = True
    
    def _compute_length_stats(self):
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
    
    def generate_data(self):
        if self._generated:
            return
            
        num_positive = int(self.num_samples * self.pos_ratio)
        num_negative = self.num_samples - num_positive
        
        # Generate positive samples
        for _ in range(num_positive):
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)
            self.data.append(sequence)
            self.labels.append(1)
        
        # Generate negative samples
        for _ in range(num_negative):
            target_length = random.randint(1, self.max_len)
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
        attention_masks = []
        
        for seq in sequences:
            padded_seq = torch.cat([
                seq,
                torch.full((max_len - len(seq),), self.pad_token_id, dtype=torch.long)
            ])
            
            attention_mask = torch.cat([
                torch.ones(len(seq), dtype=torch.long),
                torch.zeros(max_len - len(seq), dtype=torch.long)
            ])
            
            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)
        
        return {
            'input': torch.stack(padded_sequences),
            'output': torch.stack(list(labels)),
            'attention_mask': torch.stack(attention_masks)
        }


class GeneratorDataset(TrainDataset):
    """Dataset for training autoregressive language models on formal languages."""
    
    def __init__(self,
                 language,
                 num_samples,
                 max_len, 
                 seed = 42,
                 pos_only = True,
                 pos_ratio = 0.8,
                 **kwargs):
        super().__init__(language, num_samples, max_len, seed, **kwargs)
        self.pos_only = pos_only
        self.pos_ratio = pos_ratio if not pos_only else 1.0
    
    def generate_data(self):
        if self._generated:
            return
            
        if self.pos_only:
            for _ in range(self.num_samples):
                target_length = random.choice(self.PEG.valid_lengths)
                sequence = self.PEG.positive_generator(target_length)
                self.data.append(sequence)
                self.labels.append(1)
        else:
            num_positive = int(self.num_samples * self.pos_ratio)
            num_negative = self.num_samples - num_positive
            
            # Generate positive samples
            for _ in range(num_positive):
                target_length = random.choice(self.PEG.valid_lengths)
                sequence = self.PEG.positive_generator(target_length)
                self.data.append(sequence)
                self.labels.append(1)
            
            # Generate negative samples
            for _ in range(num_negative):
                target_length = random.randint(1, self.max_len)
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
        attention_masks = []
        
        for inp, tgt in zip(inputs, targets):
            inp_padded = torch.cat([
                inp, 
                torch.full((max_len - len(inp),), self.pad_token_id, dtype=torch.long)
            ])
            
            tgt_padded = torch.cat([
                tgt,
                torch.full((max_len - len(tgt),), self.pad_token_id, dtype=torch.long)
            ])
            
            attention_mask = torch.cat([
                torch.ones(len(inp), dtype=torch.long),
                torch.zeros(max_len - len(inp), dtype=torch.long)
            ])
            
            padded_inputs.append(inp_padded)
            padded_targets.append(tgt_padded)
            attention_masks.append(attention_mask)
        
        return {
            'inputs': torch.stack(padded_inputs),
            'outputs': torch.stack(padded_targets),
            'masks': torch.stack(attention_masks)
        }


def create_train_dataset(type, **kwargs):
    if type.lower() == "recognizer":
        return RecognizerDataset(**kwargs)
    elif type.lower() == "generator":
        return GeneratorDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {type}. Must be 'recognizer' or 'generator'.")
