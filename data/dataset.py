import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle as pkl
import random
from abc import ABC, abstractmethod

from .PEG import PEG


class PEGDataset(Dataset, ABC):
    """Base class for datasets on formal languages."""
    def __init__(self, language, num_iters, max_len, seed=42, **kwargs):
        self.language = language
        self.num_iters = num_iters
        self.max_len = max_len
        self.seed = seed

        self.PEG = PEG(language, max_length=max_len)
        self.pad_token = "<pad>"
        self.pad_token_id = self.PEG.stoi[self.pad_token]

        self.data = []
        self.labels = []
        self._generated = False

    @abstractmethod
    def generate_data(self, quiet=False):
        pass

    def load_data(self, path_to_results, dataset_type):
        base_dir = os.path.join(path_to_results,
                                "data", dataset_type, self.language)
        data_path = os.path.join(base_dir, "dataset.pkl")

        with open(data_path, "rb") as f:
            saved_data = pkl.load(f)
            self.data = saved_data["data"]
            self.labels = saved_data["labels"]
            self._generated = True

    def __len__(self):
        return len(self.data) if self._generated else self.num_iters


class RecognizerDataset(PEGDataset):
    """Dataset for training binary classifiers on formal languages."""

    def __init__(self,
                 language,
                 num_iters,
                 max_len,
                 seed=42,
                 pos_ratio=0.5,
                 **kwargs):
        super().__init__(language, num_iters, max_len, seed, **kwargs)
        self.pos_ratio = pos_ratio
        self._is_binary = True

    def generate_data(self, quiet=False):
        if self._generated:
            return

        num_positive = int(self.num_iters * self.pos_ratio)
        num_negative = self.num_iters - num_positive

        # Generate positive samples
        for _ in tqdm(range(num_positive),
                      desc="Generating positive samples",
                      disable=quiet):
            target_length = random.choices(
                self.PEG.valid_lengths,
                weights=self.PEG.length_weights,
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
            raise RuntimeError("Data not generated. "
                               "Call generate_data() first.")

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
                torch.full((max_len - len(seq),), self.pad_token_id,
                           dtype=torch.long)
            ])
            padded_sequences.append(padded_seq)

        return {
            'inputs': torch.stack(padded_sequences),
            'outputs': torch.stack(list(labels)),
        }


class GeneratorDataset(PEGDataset):

    def __init__(self,
                 language,
                 num_iters,
                 max_len,
                 seed=42,
                 pos_ratio=1.0,
                 **kwargs):
        super().__init__(language, num_iters, max_len, seed, **kwargs)
        self.pos_ratio = pos_ratio

    def generate_data(self, quiet=False):
        if self._generated:
            return

        if self.pos_ratio == 1.0:
            for _ in tqdm(range(self.num_iters),
                          desc="Generating positive samples",
                          disable=quiet):
                target_length = random.choices(
                    self.PEG.valid_lengths,
                    weights=self.PEG.length_weights,
                )[0]
                sequence = self.PEG.positive_generator(target_length)
                self.data.append(sequence)
                self.labels.append(1)
        else:
            num_positive = int(self.num_iters * self.pos_ratio)
            num_negative = self.num_iters - num_positive

            # Generate positive samples
            for _ in tqdm(range(num_positive),
                          desc="Generating positive samples",
                          disable=quiet):
                target_length = random.choices(
                    self.PEG.valid_lengths,
                    weights=self.PEG.length_weights,
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
            raise RuntimeError("Data not generated. "
                               "Call generate_data() first.")

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
            pad_len_inp = max_len - len(inp)
            pad_len_tgt = max_len - len(tgt)
            padded_sequence = torch.cat(
                [inp, torch.tensor([self.pad_token_id] * pad_len_inp)]
            )
            padded_target = torch.cat(
                [tgt, torch.tensor([self.pad_token_id] * pad_len_tgt)]
            )

            padded_inputs.append(padded_sequence)
            padded_targets.append(padded_target)

        return {
            'inputs': torch.stack(padded_inputs),
            'outputs': torch.stack(padded_targets),
        }


class ProbeDataset(PEGDataset):
    def generate_data(self, quiet=False):
        if self._generated:
            return

        for _ in tqdm(range(self.num_iters),
                      desc="Generating positive samples",
                      disable=quiet):
            target_length = random.choice(self.PEG.valid_lengths)
            sequence = self.PEG.positive_generator(target_length)
            label = self._generate_label(sequence)

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
    def _generate_label(self, sequence):
        return self.PEG.parse_state_generator(sequence)


class ParseDepthDataset(ProbeDataset):
    def _generate_label(self, sequence):
        return self.PEG.parse_depth_generator(sequence)


class TokenCategoryDataset(ProbeDataset):
    def __init__(self, category=None, **kwargs):
        super().__init__(**kwargs)
        self.category = category

    def _generate_label(self, sequence):
        return self.PEG.token_category_generator(sequence, self.category)
