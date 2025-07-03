import torch
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from .train_dataset import create_train_dataset
from .probe_dataset import PSDataset


def get_dataloader(lang, task, cfg, work_dir, seed=42, kind="PEG"):
    if kind == "PEG":
        dataset = create_train_dataset(
            type=task,
            language=lang,
            **OmegaConf.to_object(cfg),
            seed=seed)
    elif kind == "PS":
        dataset = PSDataset(language=lang, **OmegaConf.to_object(cfg), seed=seed)

    if cfg.precomp:
        dataset.load_data(work_dir)
    else:
        dataset.generate_data()

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=dataset.collate_fn,
    )

    return dataloader
