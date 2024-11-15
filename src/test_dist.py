import argparse, sys
import json
from data.generate import LangDataset, ALPHA
from model.transformer import Transformer, TransformerConfig
import torch
from tqdm.auto import tqdm


class Generator:

    def __init__(self, model, config, lang):
        self.alphabet = list(ALPHA[lang]) + ["<bos>", "<eos>"]
        self.stoi = {c: i for i, c in enumerate(self.alphabet)}
        self.model = model
        self.config = config

        self.model.eval()
        self.model.to(self.config["device"])

    def generate(self, prompt, max_len):
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
        prompt = prompt.to(self.config["device"])

        output = self.model.sample(prompt, max_len)
        return output

    def decode(self, tensor):
        return "".join([self.alphabet[i] for i in tensor[0].tolist()])
            

def actually_generate(generator, filename, samples):
    with open(filename, "w") as file:
        for _ in tqdm(range(samples)):
            prompt = [generator.stoi["<bos>"]]
            output = generator.generate(prompt, generator.config["N_max"])
            print(generator.decode(output), file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    with open("../config.json", "r") as f:
        config = json.load(f)
    config["t_vocab"] = len(ALPHA[args.lang]) + 2
    config["s_vocab"] = len(ALPHA[args.lang]) + 2
    
    model = Transformer(TransformerConfig.from_dict(config))
    state = torch.load(
        f"{args.work_dir}/models/models/{args.model_name}.pt",
        weights_only=True
    )
    model.load_state_dict(state)
    generator = Generator(model, config, args.lang)

    actually_generate(generator, f"{args.work_dir}/data/gens/{args.lang}.txt", 1)
