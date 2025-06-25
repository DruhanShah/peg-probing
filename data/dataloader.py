import torch
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from .train_dataset import PEGDataset
from .probe_dataset import PSDataset


def get_dataloader(lang, cfg, work_dir, seed=42, kind="PEG"):
    if kind == "PEG":
        dataset = PEGDataset(language=lang, **OmegaConf.to_object(cfg), seed=seed)
    elif kind == "PS":
        dataset = PSDataset(language=lang, **OmegaConf.to_object(cfg), seed=seed)

    if cfg.precomp:
        dataset.load_data(work_dir)

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    padded_sequences = []
    padded_masks = []
    padded_labels = []

    max_len = max(seq.size(0) for seq in sequences)
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padded_seq = torch.cat([seq,
                                torch.zeros(pad_len, dtype=seq.dtype)])
        padded_mask = torch.cat([torch.ones(seq.size(0), dtype=seq.dtype),
                                    torch.zeros(pad_len, dtype=seq.dtype)])
        padded_sequences.append(padded_seq)
        padded_masks.append(padded_mask)

    if labels[0].shape != torch.Size([]):
        max_len = max(label.size(0) for label in labels)
        for label in labels:
            pad_len = max_len - label.size(0)
            padded_label = torch.cat([label,
                                      torch.full(tuple([pad_len]), label[-1])])
            padded_labels.append(padded_label)
    else:
        padded_labels = labels

    sequences_tensor = torch.stack(padded_sequences)
    masks_tensor = torch.stack(padded_masks)
    labels_tensor = torch.stack(padded_labels)

    return {
        "input_ids": sequences_tensor,
        "masks": masks_tensor,
        "labels": labels_tensor
    }
