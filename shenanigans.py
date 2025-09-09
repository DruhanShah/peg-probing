import hydra
import torch

from data import get_dataloader
from model import RecognizerModel, GeneratorModel

from utils import init_wandb, set_seed, open_log, cleanup, visualise

FP = None


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, ["shenanigans"])
    set_seed(cfg.seed)
    FP = open_log(cfg)

    dataloader = get_dataloader(
        cfg.lang, cfg.probe.type,
        cfg.shenanigans,
        cfg.work_dir, cfg.seed,
        kind="probe", quiet=(not cfg.deploy),
    )

    # Load the main model
    model_path = (f"{cfg.work_dir}/models/{cfg.model.type}/"
                  f"{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt")
    model_state = torch.load(model_path, weights_only=False)
    model_state["config"].act_cache = cfg.model.act_cache
    model = (RecognizerModel(model_state["config"])
             if cfg.model.type == "recognizer"
             else GeneratorModel(model_state["config"]))
    model.load_state_dict(model_state["net"])

    for batch in dataloader:
        _in = batch["inputs"].to(cfg.device)
        _out = model(_in, return_type=["logits", "cache"])
        _, _cache = _out["logits"], _out["cache"]

        string = _in[0][1:-1].cpu().numpy()
        string = dataloader.dataset.PEG.detokenize_string(string)

        map = _cache["block_0.attn_map"][0, :, :, :].detach()
        map = map.to(dtype=torch.float32).cpu().numpy()
        visualise(map, "Layer 0", string)
        map = _cache["block_1.attn_map"][0, :, :, :].detach()
        map = map.to(dtype=torch.float32).cpu().numpy()
        visualise(map, "Layer 1", string)

    FP = cleanup(cfg, FP)


if __name__ == "__main__":
    main()
