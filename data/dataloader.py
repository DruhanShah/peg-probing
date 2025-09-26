from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from .dataset import DATASETS


def get_dataloader(lang, task, cfg, work_dir,
                   seed=42, kind="model", quiet=False):

    dataset = DATASETS[kind][task](lang, **OmegaConf.to_object(cfg), seed=seed)

    if cfg.precomp:
        dataset.load_data(work_dir, task)
    else:
        dataset.generate_data(quiet=quiet)

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
