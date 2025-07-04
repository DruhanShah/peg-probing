import random
from itertools import cycle
from parsimonious.grammar import Grammar
from parsimonious.exceptions import IncompleteParseError, ParseError

from data.strings import gen_triple, gen_star, gen_dyck_1, gen_dyck_2, gen_expr


GRAMMAR = {
    "star": """
        S = ~"a*b*"
        """,
    "dyck-1": """
        S = "(" T ")" T
        T = S / ""
        """,
    "dyck-2": """
        S = A / B
        A = "(" T ")" T
        B = "[" T "]" T
        T = S / ""
        """,
    "dyck-3": """
        S = A / B / C
        A = "(" T ")" T
        B = "[" T "]" T
        C = "{" T "}" T
        T = S / ""
        """,
    "triple": """
        S = &(A "c") ~"a+" B
        A = "a" A? "b"
        B = "b" B? "c"
    """,
    "expr": """
        S = E / D
        E = O S S
        O = ~"[+*^]"
        D = ~"[0-9]"
    """,
}

ALPHABET = {
        "star": list("ab"),
        "dyck-1": list("()"),
        "dyck-2": list("()[]"),
        "dyck-3": list("()[]{}"),
        "triple": list("abc"),
        "expr": list("0123456789+*^"),
}

FUNCS = {
    "star": gen_star,
    "dyck-1": gen_dyck_1,
    "dyck-2": gen_dyck_2,
    "triple": gen_triple,
    "expr": gen_expr,
}


class PEG:

    def __init__(self, language, max_length=30):
        self.language = language
        self.alphabet = ["<eos>"] + ALPHABET[language]
        self.grammar = Grammar(GRAMMAR[language])
        self.max_length = max_length

        self.vocab_size = len(self.alphabet)
        self.stoi = {char: i for i, char in enumerate(self.alphabet)}
        self.itos = {i: char for i, char in enumerate(self.alphabet)}
        if self.language == "star":
            self.valid_lengths = list(range(1, self.max_length+1))
        elif self.language == "dyck-1":
            self.valid_lengths = list(range(2, self.max_length+1, 2))
        elif self.language == "dyck-2":
            self.valid_lengths = list(range(2, self.max_length+1, 2))
        elif self.language == "triple":
            self.valid_lengths = list(range(3, self.max_length+1, 3))
        elif self.language == "expr":
            self.valid_lengths = list(range(1, self.max_length+1, 2))

    def tokenize_string(self, string):
        tokens = list(string) + ["<eos>"]
        token_indices = [self.stoi[token] for token in tokens]
        return token_indices

    def detokenize_string(self, token_indices):
        tokens = [self.itos[token] for token in token_indices.tolist()]
        return "".join(tokens)

    def grammar_check(self, string):
        try:
            self.grammar.parse(string)
            return True
        except (IncompleteParseError, ParseError):
            return False

    def positive_generator(self, length):
        if length not in self.valid_lengths:
            raise ValueError(f"Invalid string length for {self.language}: {length}")

        langfunc = FUNCS[self.language]

        positive_string = langfunc(length)
        assert self.grammar_check(positive_string), \
            f"Generated string is not valid: {positive_string}"
        return positive_string

    def negative_generator(self, length):
        alphabet = [i for i in self.alphabet if i not in ["<eos>"]]
        negative_string = ''.join(random.choices(alphabet, k=length))
        
        max_attempts = 40
        attempts = 0
        while self.grammar_check(negative_string) and attempts < max_attempts:
            idx = random.randint(0, len(negative_string) - 1)
            negative_string = (negative_string[:idx] + 
                                random.choice(alphabet) + 
                                negative_string[idx+1:])
            attempts += 1
        
        assert not self.grammar_check(negative_string), \
            f"Generated negative string is valid: {negative_string}"
        return negative_string

    def parse_state_generator(self, string):
        state = []
        for i in range(len(string)):
            state.append(self.grammar_check(string[:i+1]))
        state.append(state[-1])  # Account for the first <eos> token
        return state

