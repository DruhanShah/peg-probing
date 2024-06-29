from parsimonious.grammar import Grammar
from parsimonious.exceptions import IncompleteParseError, ParseError
from tqdm import tqdm
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
        S = A / ""
        A = "(" S ")" S
    """),
    "dyck2": Grammar("""
        S = A / B / ""
        A = "(" S ")" S
        B = "[" S "]" S
    """),
    "dyck3": Grammar("""
        S = A / B / C / ""
        A = "(" S ")" S
        B = "[" S "]" S
        C = "{" S "}" S
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


def verify(sample, grammar):
    try:
        grammar.parse(sample)
        return True
    except IncompleteParseError:
        return False
    except ParseError:
        return False


def generate_random(samples, lang):
    alphabet = ALPHA[lang]
    final = 0
    with open(f"../data/{lang}.txt", "w") as f:
        with tqdm(total=samples) as progress:
            while final < samples:
                n = random.randint(1, MAX_LENGTH)
                s = "".join(random.choice(alphabet) for _ in range(n))
                if verify(s, GRAMMAR[lang]):
                    progress.update(1)
                    final += 1
                    print(s, file=f)
    print(f"Finished {lang}")


if __name__ == "__main__":
    for lang in LANGS:
        generate_random(100000, lang)
