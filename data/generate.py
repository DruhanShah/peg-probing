import argparse, sys
from parsimonious.grammar import Grammar
from parsimonious.exceptions import IncompleteParseError, ParseError
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import random

MAX_LENGTH = 30
LANGS = ["triple", "star", "brack", "dyck1", "dyck2", "dyck3", "expr"]
ALPHA = {
        "triple": "abc",
        "star": "ab",
        "brack": "ab[]",
        "dyck1": "()",
        "dyck2": "()[]",
        "dyck3": "()[]{}",
        "expr": "()+*^0123456789",
        }
GRAMMAR = {
        "triple": Grammar("""
                          S = &(A "c") ~"a+" B
                          A = "a" A? "b"
                          B = "b" B? "c"
                          """),
        "star": Grammar("""
                        S = ~"a*b*"
                        """),
        "brack": Grammar("""
                         S = "[" T "]"
                         T =  T1 / T2 / T3
                         T1 = "a" S "b"
                         T2 = ~"a*b"
                         T3 = ~"ab*"
                         """),
        "dyck1": Grammar("""
                         S = "(" T ")" T
                         T = S / ""
                         """),
        "dyck2": Grammar("""
                         S = A / B
                         A = "(" T ")" T
                         B = "[" T "]" T
                         T = S / ""
                         """),
        "dyck3": Grammar("""
                         S = A / B / C
                         A = "(" T ")" T
                         B = "[" T "]" T
                         C = "{" T "}" T
                         T = S / ""
                         """),
        "expr": Grammar("""
                        S = M ("+" M)*
                        M = E ("*" E)*
                        E = V ("^" V)*
                        V = D / C
                        C = "(" S ")"
                        D = ~"[0-9]"
                        """),
        }


class LangDataset(Dataset):

    def __init__(self, lang, samples, device):
        self.samples = samples
        self.grammar = GRAMMAR[lang]
        self.device = device
        self.alpha = list(ALPHA[lang]) + ["<bos>", "<eos>"]
        self.stoi = {c: i for i, c in enumerate(self.alpha)}

    def __getitem__(self, index):
        s = ["<bos>"] + list(self.samples[index]) + ["<eos>"]
        tense = [self.stoi[c] for c in s]
        return torch.tensor(tense, dtype=torch.long).to(self.device)

    def __len__(self):
        return len(self.samples)


class Generator:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        pass

    def verify(self, sample, grammar):
        try:
            grammar.parse(sample)
        except IncompleteParseError:
            return True
        except:
            return False
        else:
            return True

    def generate_random(self, samples, lang):
        save_dir = sys.argv[1]
        alphabet = ALPHA[lang]
        final = 0
        with open(f"{self.save_dir}/data/corpus/{lang}.txt", "w") as file:
            with tqdm(total=samples) as progress:
                while final < samples:
                    s = "".join(random.choice(alphabet) for _ in range(MAX_LENGTH))
                    if self.verify(s, GRAMMAR[lang]):
                        progress.update(1)
                        final += 1
                        print(s, file=file)
        print(f"Finished {lang}")

    def return_datasets(self, samples):
        for lang in LANGS:
            self.generate_random(samples, lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", help="Directory to save the generated data")
    args = parser.parse_args()
    
    gen = Generator(args.work_dir)
    gen.return_datasets(1e6)
