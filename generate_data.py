import hydra
import torch

from data import create_train_dataset
from utils import init_wandb, set_seed, cleanup
from utils import save_data
from utils import open_log, log_gen

@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, ["data_gen"])
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = create_train_dataset(
        type = cfg.model_type,
        language = cfg.lang,
        precomp = cfg.data.precomp,
        num_samples = cfg.data.num_samples,
        max_len = cfg.data.max_len,
        pos_ratio = cfg.data.pos_ratio,
        seed = cfg.seed,
    )
    dataset.generate_data()
    save_data(cfg, dataset)

    data_stats = dataset.length_stats()
    data_stats = log_gen(cfg.deploy, data_stats)
    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
