import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from data.utils import gen_triple, gen_star, gen_dyck_1, gen_dyck_2, gen_expr

from parsimonious.grammar import Grammar
from parsimonious.exceptions import IncompleteParseError, ParseError


GRAMMAR = {
    "triple": """
        S = &(A "c") ~"a+" B
        A = "a" A? "b"
        B = "b" B? "c"
        """,
    "star": """
        S = ~"a*b*"
        """,
    "brack": """
        S = "[" T "]"
        T =  T1 / T2 / T3
        T1 = "a" S "b"
        T2 = ~"a*b"
        T3 = ~"ab*"
        """,
    "dyck1": """
        S = "(" T ")" T
        T = S / ""
        """,
    "dyck2": """
        S = A / B
        A = "(" T ")" T
        B = "[" T "]" T
        T = S / ""
        """,
    "dyck3": """
        S = A / B / C
        A = "(" T ")" T
        B = "[" T "]" T
        C = "{" T "}" T
        T = S / ""
        """,
    "expr": """
        S = M ("+" M)*
        M = E ("*" E)*
        E = V ("^" V)*
        V = D / C
        C = "(" S ")"
        D = ~"[0-9]"
        """,
}

ALPHABET = {
        "triple": list("abc"),
        "star": list("ab"),
        "brack": list("ab[]"),
        "dyck1": list("()"),
        "dyck2": list("()[]"),
        "dyck3": list("()[]{}"),
        "expr": list("0123456789()+*^"),
}


class PEG:

    def __init__(self, language, max_length=30):
        self.language = language
        self.alphabet = ["<bos>", "<eos>", "<pad>"] + ALPHABET[language]
        self.grammar = Grammar(GRAMMAR[language])
        self.max_length = max_length

        self.vocab_size = len(self.alphabet)
        self.stoi = {char: i for i, char in enumerate(self.alphabet)}
        self.itos = {i: char for i, char in enumerate(self.alphabet)}


    def tokenize_string(self, string):
        tokens = ["<bos>"] + list(string) + ["<eos>"]
        token_indices = [self.stoi[token] for token in tokens]
        return token_indices

    def detokenize_string(self, token_indices):
        tokens = [self.itos[token] for token in token_indices.tolist()]
        return "".join(tokens)

    def count_prefix(self, string):
        prefix = -1
        for i in range(len(string)):
            try:
                self.grammar.parse(string[:i+1])
                prefix = i+1
            except (IncompleteParseError, ParseError):
                continue
        return prefix

    def check_grammaticality(self, string):
        pref = self.count_prefix(string)
        return (pref > -1), pref

    def string_generator(self, samples):
        funcs = {
            "triple": gen_triple,
            "star": gen_star,
            "dyck-1": gen_dyck_1,
            "dyck-2": gen_dyck_2,
            "expr": gen_expr,
        }

        valid_lengths = {
            "triple": list(range(3, 61, 3)),
            "dyck-1": list(range(2, 63, 2)),
            "dyck-2": list(range(2, 63, 2)),
            "star": list(range(1, 63)),
            "expr": list(range(1, 63)),
        }

        if self.lang not in funcs:
            raise ValueError("Invalid language")

        langfunc = funcs[self.lang]
        langlen = valid_lengths[self.lang]

        selected_lengths = []
        while len(selected_lengths) < samples:
            selected_lengths.extend(langlen)
        selected_lengths = selected_lengths[:samples]

        random.shuffle(selected_lengths)
        for l in selected_lengths:
            output = langfunc(l)
            output += random.choices(self.alphabet[3:], k=self.max_length - l)
            yield output
