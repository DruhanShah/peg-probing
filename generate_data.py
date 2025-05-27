import hydra
import torch

from data import PEGDataset
from utils import init_wandb, set_seed, cleanup
from utils import open_log, log_gen

@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = PEGDataset(
        language=cfg.data.language,
        precomp=cfg.data.precomp,
        num_iters=cfg.data.num_iters,
        max_len=cfg.data.max_len,
        seed=cfg.seed,
    )

    data_stats = dataset.save_data(cfg.work_dir, cfg.data.num_iters)
    log_gen(cfg.deploy, data_stats)
    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
