import random

def gen_triple(n):
    if n % 3: return ""
    m = n // 3
    return "a" * m + "b" * m + "c" * m


def gen_star(n):
    m = random.randint(0, n)
    return "a" * m + "b" * (n - m)


def gen_dyck_1(n):
    if n == 0: return ""
    if n == 2: return "()"
    
    s = 2*random.randint(1, n//2-1)
    return (gen_dyck_1(s) + gen_dyck_1(n-s)
            if random.random() < 0.5
            else "(" + gen_dyck_1(n-2) + ")")


def gen_dyck_2(n):
    if n == 0: return ""
    if n == 2: return random.choice(["()", "[]"])
    
    s = 2*random.randint(1, n//2-1)
    o, c = tuple(random.choice(["()", "[]"]))
    return (gen_dyck_2(s) + gen_dyck_2(n-s)
            if random.random() < 0.5
            else o + gen_dyck_2(n-2) + c)


def gen_expr(n):
    if n == 0: return ""

    operators = list("+*^")
    operands = list("0123456789")

    root = ["operand", random.choice(operands)]
    leaves = [root]
    while n > 1:
        chosen = random.choice(leaves)
        leaves.remove(chosen)

        left = ["operand", random.choice(operands)]
        right = ["operand", random.choice(operands)]
        chosen[0] = "operator"
        chosen[1] = random.choice(operators)
        chosen += [left, right]

        leaves += [left, right]
        n -= 2

    def preorder(node):
        return (node[1] + preorder(node[2]) + preorder(node[3])
                if node[0] == "operator"
                else node[1])

    return preorder(root)
