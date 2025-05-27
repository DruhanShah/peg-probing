import random

LEAF = 0
BRANCH = 1

class ExprNode:

    def __init__(self):
        self.cat = LEAF

    def branch(self, c1, c2):
        self.cat = BRANCH
        self.left = c1
        self.right = c2


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

    def preorder(node):
        return (random.choice(operators)
                + preorder(node.left)
                + preorder(node.right)
                if node.cat == BRANCH
                else random.choice(operands))

    root = ExprNode()
    leaves = [root]
    for _ in range(n//2):
        chosen = random.choice(leaves)
        leaves.remove(chosen)

        left = ExprNode()
        right = ExprNode()
        chosen.branch(left, right)
        leaves += [left, right]

    result = preorder(root)
    return result
