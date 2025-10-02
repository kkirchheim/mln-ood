import logging
from typing import List, Any

log = logging.getLogger(__name__)


class Node:
    def __init__(self):
        self._children: List["Node"] = []

    def size(self):
        return 1 + sum([c.size() for c in self._children])

    def __contains__(self, item):
        if item.__class__ == self.__class__:
            return True
        for child in self._children:
            if child and item in child:
                return True
        return False

    def is_leaf(self):
        return len(self._children) == 0

    def __iter__(self):
        yield self
        for child in self._children:
            yield from child


class Operator(Node):
    def is_same(self, other):
        return other.__class__ == self.__class__


class Atom(Node):
    pass


class BinaryVariable(Atom):
    """Binary attribute, interpreted as x[:, IDX_ATTR] == 1."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return isinstance(other, BinaryVariable) and self.name == other.name


class CategoricalVariable(Atom):
    """
    Categorical attribute, AST stores the VALUE AS STRING TOKEN
    e.g., value_str = "red" or "1" (string). Integer mapping is deferred to codegen.
    """
    def __init__(self, name: str, value_str: str):
        super().__init__()
        self.name = name
        self.value_str = value_str  # keep as human-readable string token

    def __repr__(self):
        return f"{self.name}={self.value_str}"

    def __eq__(self, other):
        return (
            isinstance(other, CategoricalVariable)
            and self.name == other.name
            and self.value_str == other.value_str
        )


class UnaryOperator(Operator):
    def __init__(self, child=None):
        super().__init__()
        self._children = []
        self._child = None
        self.child = child

    @property
    def child(self):
        return self._child

    @child.setter
    def child(self, value):
        if self._child in self._children:
            self._children.remove(self._child)
        self._child = value
        if value:
            self._children.append(value)


class BinaryOperator(Operator):
    def __init__(self, left=None, right=None):
        super().__init__()
        self._children = []
        self._left = None
        self._right = None
        self.left = left
        self.right = right

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        if self._left in self._children:
            self._children.remove(self._left)
        self._left = value
        if value:
            self._children.append(value)

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        if self._right in self._children:
            self._children.remove(self._right)
        self._right = value
        if value:
            self._children.append(value)


class Not(UnaryOperator):
    def __init__(self, expr):
        super().__init__(expr)
        if not isinstance(expr, (BinaryVariable, CategoricalVariable)):
            raise ValueError("Negation can only be applied to a variable-level proposition")

    def __repr__(self):
        return f"not {self.child}"


class And(BinaryOperator):
    def __repr__(self):
        return f"({self.left} and {self.right})"


class Or(BinaryOperator):
    def __repr__(self):
        return f"({self.left} or {self.right})"


class Xor(BinaryOperator):
    def __repr__(self):
        return f"({self.left} xor {self.right})"


class Implies(BinaryOperator):
    def __repr__(self):
        return f"({self.left} -> {self.right})"


class ConstraintCompiler:
    def __init__(self, attribute_index_map, category_value_map=None):
        self.attribute_index_map = attribute_index_map
        self.category_value_map = category_value_map if category_value_map is not None else {}
        self.variables = list(attribute_index_map.keys())
        self.operators = ["and", "or", "not", "->", "xor", "either", "(", ")", "="]
        # Allow structured keys: str for binary vars, tuple (attr, value_str) for categorical vars
        self.var_mappings: dict[Any, str] = {}

    def tokenize(self, s):
        s = s.replace("->", " -> ").replace("(", " ( ").replace(")", " ) ").replace("=", " = ")
        return s.lower().split()

    class Parser:
        def __init__(self, tokens, variables, operators, compiler):
            self.tokens = tokens
            self.pos = 0
            self.variables = variables
            self.operators = operators
            self.compiler = compiler

        def peek_token(self):
            return self.tokens[self.pos] if self.pos < len(self.tokens) else None

        def consume_token(self, expected_token=None):
            if self.pos >= len(self.tokens):
                return None
            token = self.tokens[self.pos]
            if expected_token is not None and token != expected_token:
                raise ValueError(f"Expected token '{expected_token}', got '{token}'")
            self.pos += 1
            return token

        def parse_expression(self) -> Node:
            if self.peek_token() == "either":
                self.consume_token("either")
            return self.parse_implies()

        def parse_implies(self):
            node = self.parse_or()
            while self.peek_token() == "->":
                self.consume_token("->")
                right = self.parse_or()
                node = Implies(node, right)
            return node

        def parse_or(self):
            node = self.parse_and()
            while self.peek_token() in ("or", "xor"):
                op = self.consume_token()
                right = self.parse_and()
                node = Or(node, right) if op == "or" else Xor(node, right)
            return node

        def parse_and(self):
            node = self.parse_not()
            while self.peek_token() == "and":
                self.consume_token("and")
                right = self.parse_not()
                node = And(node, right)
            return node

        def parse_not(self):
            if self.peek_token() == "not":
                self.consume_token("not")
                expr = self.parse_atom()
                return Not(expr)
            return self.parse_atom()

        def parse_atom(self):
            token = self.peek_token()
            if token == "(":
                self.consume_token("(")
                expr = self.parse_expression()
                if self.peek_token() != ")":
                    raise ValueError("Expected ')'")
                self.consume_token(")")
                return expr
            return self.parse_variable()

        def parse_variable(self):
            attr_token = self.consume_token()
            if attr_token not in self.variables:
                raise ValueError(f"Unknown attribute '{attr_token}'")

            if self.peek_token() == "=":
                self.consume_token("=")
                value_token = self.consume_token()
                # Validate, but KEEP the original string token in the AST.
                self._validate_value(attr_token, value_token)
                return CategoricalVariable(attr_token, value_token)
            else:
                if self._is_binary(attr_token):
                    return BinaryVariable(attr_token)
                raise ValueError(f"Attribute '{attr_token}' is multi-valued; please specify '{attr_token}=...'")

        def _is_binary(self, attr):
            if attr not in self.compiler.category_value_map:
                return True
            return len(self.compiler.category_value_map[attr]) == 2

        def _validate_value(self, attr, value_token):
            # Categorical attr: accept digit or known category name
            if attr in self.compiler.category_value_map:
                if value_token.isdigit():
                    return
                if value_token in self.compiler.category_value_map[attr]:
                    return
                raise ValueError(f"Unknown category '{value_token}' for attribute '{attr}'")
            # Non-categorical: require a digit
            if not value_token.isdigit():
                raise ValueError(
                    f"Attribute '{attr}' is not categorical, use a digit or shorthand '{attr}' for =1."
                )

    # -------- code generation helpers --------

    def get_var_name(self, key: Any):
        """
        key is either:
          - str: binary variable name, e.g., "roadwork"
          - tuple(attr, value_str): e.g., ("color", "red") or ("class", "1")
        Returns a stable Python identifier string.
        """
        if key not in self.var_mappings:
            if isinstance(key, tuple):
                attr, value_str = key
                # Build an identifier safe name. Keep value string for readability.
                code_var_name = f"{attr}__eq__{value_str}"
            else:
                code_var_name = key
            self.var_mappings[key] = code_var_name
        return self.var_mappings[key]

    def _resolve_value_to_int(self, attr: str, value_str: str) -> int:
        """
        Map the human-readable value string to the integer used in x[:, IDX_ATTR].
        - If categorical and value_str is name: look up in category_value_map[attr].
        - If value_str is digit: int(value_str).
        """
        if value_str.isdigit():
            return int(value_str)
        if attr in self.category_value_map:
            cat_map = self.category_value_map[attr]
            if value_str in cat_map:
                return cat_map[value_str]
        # If not resolvable, raise: codegen cannot proceed.
        raise ValueError(f"Cannot map '{attr}={value_str}' to an integer value")

    def collect_definitions(self):
        definitions = []
        for key, code_var_name in self.var_mappings.items():
            # Binary variable
            if isinstance(key, str):
                if key not in self.attribute_index_map:
                    raise ValueError(f"Unknown binary attribute '{key}' in attribute_index_map")
                idx_name = self.attribute_index_map[key]
                definitions.append(f"{code_var_name} = x[:, {idx_name}] == 1")
            # Categorical variable
            else:
                attr, value_str = key
                if attr not in self.attribute_index_map:
                    raise ValueError(f"Unknown attribute '{attr}' in attribute_index_map")
                idx_name = self.attribute_index_map[attr]
                val_int = self._resolve_value_to_int(attr, value_str)
                definitions.append(f"{code_var_name} = x[:, {idx_name}] == {val_int}")
        return definitions

    def generate_constraint_expression(self, node: Node) -> str:
        if isinstance(node, BinaryVariable):
            return self.get_var_name(node.name)
        elif isinstance(node, CategoricalVariable):
            return self.get_var_name((node.name, node.value_str))
        elif isinstance(node, Not):
            expr = self.generate_constraint_expression(node.child)
            return f"~({expr})"
        elif isinstance(node, And):
            l = self.generate_constraint_expression(node.left)
            r = self.generate_constraint_expression(node.right)
            return f"({l}) & ({r})"
        elif isinstance(node, Or):
            l = self.generate_constraint_expression(node.left)
            r = self.generate_constraint_expression(node.right)
            return f"({l}) | ({r})"
        elif isinstance(node, Xor):
            l = self.generate_constraint_expression(node.left)
            r = self.generate_constraint_expression(node.right)
            return f"({l}) ^ ({r})"
        elif isinstance(node, Implies):
            l = self.generate_constraint_expression(node.left)
            r = self.generate_constraint_expression(node.right)
            return f"(~({l})) | ({r})"
        else:
            raise ValueError("Unknown node type")

    def generate_satisfaction_condition(self, node):
        return self.generate_constraint_expression(node)

    def compile(self, function_name, input_string):
        log.debug(f"Compiling '{input_string}'")
        tokens = self.tokenize(input_string)
        parser = ConstraintCompiler.Parser(tokens, self.variables, self.operators, self)
        ast = parser.parse_expression()

        self.var_mappings.clear()
        condition_expr = self.generate_satisfaction_condition(ast)
        var_defs = self.collect_definitions()

        code_lines = [f"def {function_name}(x):"]
        for line in var_defs:
            code_lines.append(f"    {line}")
        code_lines.append(f"    return torch.where({condition_expr}, 1, 0).unsqueeze(1)")
        return "\n".join(code_lines)

# Example usage
#
if __name__ == "__main__":
    # Example attribute-index map
    attribute_index_map = {
        "color": "IDX_COLOR",
        "class": "IDX_CLASS",
        "shape": "IDX_SHAPE",
    }

    # Suppose color, class, shape are categorical
    category_value_map = {
        "color": {"red": 0, "blue": 1},
        "class": {"stop_sign": 1, "yield_sign": 2},
        "shape": {"octagon": 1, "circle": 0},
    }

    compiler = ConstraintCompiler(attribute_index_map, category_value_map)

    # # Example 1: A multi-valued attribute, must specify =value
    # # "class=stop_sign -> shape=octagon and color=red"
    # constraint_str1 = "class=stop_sign -> (shape=octagon and color=red)"
    # code1 = compiler.compile("stop_sign_constraint", constraint_str1)
    # print("\n--- Example 1 ---")
    # print(constraint_str1, "=>\n", code1)
    #
    # # Example 2: A purely binary attribute not in category_value_map:
    # attribute_index_map["roadwork"] = "IDX_ROADWORK"
    # # Not in category_value_map => "roadwork" can be used as "roadwork=1" or shorthand "roadwork"
    # compiler.variables.append("roadwork")  # Add to recognized attributes
    #
    # constraint_str2 = "roadwork -> shape=octagon"
    # code2 = compiler.compile("roadwork_constraint", constraint_str2)
    # print("\n--- Example 2 ---")
    # print(constraint_str2, "=>\n", code2)
    #
    # # Example 3: If "color" has exactly 2 possible values, we can do "color -> shape=octagon"
    # # That means "color=1 -> shape=octagon"? We interpret "color" as "color=1"
    # constraint_str3 = "color -> shape=octagon"
    # code3 = compiler.compile("color_octagon_constraint", constraint_str3)
    # print("\n--- Example 3 ---")
    # print(constraint_str3, "=>\n", code3)
    tokens = compiler.tokenize("class=stop_sign and color=blue")
    variables = list(attribute_index_map.keys())

    # Operators now include '=' for categorical checks
    operators = ["and", "or", "not", "->", "xor", "either", "(", ")", "="]

    parser = ConstraintCompiler.Parser(tokens, variables, operators, compiler)
    ast: Node = parser.parse_expression()

    print(ast)
    print(type(ast))

    code = compiler.compile("abc", "class=stop_sign and color=red")
    print(code)
    # print(And() in ast)
    #
    # print(Xor() in ast)
