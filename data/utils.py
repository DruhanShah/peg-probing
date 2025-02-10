import random

def gen_triple(length):
    if length % 3: return ""
    n = length // 3
    return "a" * n + "b" * n + "c" * n


def gen_star(length):
    n = random.randint(0, length)
    return "a" * n + "b" * (length - n)


def gen_dyck_1(length):
    if length % 2: return ""
    s, o, c = [], 0, 0
    for _ in range(length):
        if o < length // 2 and (c == o or random.choice([True, False])):
            s.append("(")
            o += 1
        else:
            s.append(")")
            c += 1
    return "".join(s)


def gen_dyck_2(length):
    if length % 2: return ""
    s, o, c = [], [0, 0], [0, 0]
    for _ in range(length):
        ch = random.choice([(o[0] < length // 2, "("), (o[1] < length // 2, "["), 
                            (c[0] < o[0] and s[-1:] == ["("], ")"), 
                            (c[1] < o[1] and s[-1:] == ["["], "]")])
        if ch[0]: 
            s.append(ch[1])
            o[0] += ch[1] == "("
            o[1] += ch[1] == "["
            c[0] += ch[1] == ")"
            c[1] += ch[1] == "]"
    return "".join(s)


def gen_expr(length):
    if length < 1: return ""
    if length == 1: return str(random.randint(1, 9))  # Single operand
    ops = ["+", "-", "*", "/"]
    left_length = random.randint(1, length - 2)  # Split length between operands
    return random.choice(ops) + rand_prefix_expr(left_length) + rand_prefix_expr(length - 1 - left_length)

