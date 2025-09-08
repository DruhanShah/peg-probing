import random
from parsimonious import Grammar, NodeVisitor

LEAF = 0
BRANCH = 1


class DepthCalculator(NodeVisitor):
    def __init__(self, grammar):
        self.grammar = grammar
        self.current_depth = 0
        self.depths = []

    def get_depths(self, string):
        self.depths = [0] * len(string)
        self.current_depth = 0
        tree = self.grammar.parse(string)
        self.visit(tree)
        return self.depths

    def generic_visit(self, node, visited_children):
        # Calculate depth for this node's text span
        start_pos = node.start
        end_pos = node.end

        # Only process nodes that actually consume text
        if start_pos < end_pos:
            # Update depths for all characters covered by this node
            for i in range(start_pos, end_pos):
                self.depths[i] = max(self.depths[i], self.current_depth)

        # Recursively visit children with increased depth
        start = hasattr(node.expr, "name") and node.expr_name == "S"
        self.current_depth += 1 if start else 0
        result = []
        for child in node:
            if hasattr(child, 'start'):
                result.append(self.visit(child))
            else:
                result.append(child)
        self.current_depth -= 1 if start else 0

        return result


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
