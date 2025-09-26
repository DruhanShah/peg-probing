import random
from parsimonious.grammar import Grammar
from parsimonious.exceptions import IncompleteParseError, ParseError

from .utils import gen_star, gen_dyck_1, gen_dyck_2, gen_expr
from .utils import DepthCalculator


GRAMMAR = {
    "star": """
        S = A B
        A = a*
        B = b*
        a = "a"
        b = "b"
        """,
    "dyck-1": """
        S = T* / ""
        T = A
        A = "(" S ")"
        """,
    "dyck-2": """
        S = T* / ""
        T = A / B
        A = "(" S ")"
        B = "[" S "]"
        """,
    "dyck-3": """
        S = T* / ""
        T = A / B / C
        A = "(" S ")"
        B = "[" S "]"
        C = "{" S "}"
        """,
    "expr": """
        S = E / D
        E = O S S
        O = "^" / "+" / "*"
        D = "0" / "1" / "2" / "3" / "4" / "5" / "6" / "7" / "8" / "9"
    """,
}

ALPHABET = {
        "star": list("ab"),
        "dyck-1": list("()"),
        "dyck-2": list("()[]"),
        "dyck-3": list("()[]{}"),
        "expr": list("0123456789+*^"),
}

FUNCS = {
    "star": gen_star,
    "dyck-1": gen_dyck_1,
    "dyck-2": gen_dyck_2,
    "expr": gen_expr,
}
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


class PEG:

    def __init__(self, language, max_length=30):
        self.language = language
        self.alphabet = SPECIAL_TOKENS + ALPHABET[language]
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
        elif self.language == "expr":
            self.valid_lengths = list(range(1, self.max_length+1, 2))
        self.length_weights = [(i+1) for i in range(len(self.valid_lengths))]

        self.depth_calculator = DepthCalculator(self.grammar)

    def tokenize_string(self, string):
        tokens = ["<bos>"] + list(string) + ["<eos>"]
        token_indices = [self.stoi[token] for token in tokens]
        return token_indices

    def detokenize_string(self, token_indices, clean=False):
        tokens = []
        start = -1
        end = -1
        for i, idx in enumerate(token_indices):
            token = self.itos[idx]
            tokens.append(token)
            if token == "<bos>" and start == -1:
                start = i
            elif token in ["<eos>", "<pad>"] and end == -1:
                end = i
        return "".join(tokens[start+1:end] if clean else tokens)

    def grammar_check(self, string):
        try:
            self.grammar.parse(string)
            return True
        except (IncompleteParseError, ParseError):
            return False

    def positive_generator(self, length):
        if length not in self.valid_lengths:
            raise ValueError(f"Invalid length for {self.language}: {length}")

        langfunc = FUNCS[self.language]

        positive_string = langfunc(length)
        assert self.grammar_check(positive_string), \
            f"Generated string is not valid: {positive_string}"
        return positive_string

    def negative_generator(self, length):
        alphabet = [i for i in self.alphabet if i not in SPECIAL_TOKENS]
        negative_string = ''.join(random.choices(alphabet, k=length))

        while self.grammar_check(negative_string):
            idx = random.randint(0, len(negative_string) - 1)
            negative_string = (negative_string[:idx] +
                               random.choice(alphabet) +
                               negative_string[idx+1:])

        assert not self.grammar_check(negative_string), \
            f"Generated negative string is valid: {negative_string}"
        return negative_string

    def parse_state_generator(self, string):
        # Important to include the states for "" and "<bos>"
        # since there's causal masking!
        state = [False]
        for i in range(len(string)+1):
            state.append(self.grammar_check(string[:i]))
        return state

    def parse_depth_generator(self, string):
        # Important to include the states for "" and "<bos>"
        # since there's causal masking!
        depths = self.depth_calculator.get_depths(string)
        return [0, 0] + depths
