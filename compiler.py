import logging
from typing import List

log = logging.getLogger(__name__)


class Node:
    def __init__(self):
        self._children: List[Node] = []

    def size(self):
        """
        return sum of number of children
        """
        return 1 + sum([c.size() for c in self._children])

    def __contains__(self, item):
        # TODO: not the best idea
        if item.__class__ == self.__class__:
            return True

        for child in self._children:
            if child and item in child:
                return True

        return False

    def is_leaf(self):
        return len(self._children) == 0

    def __iter__(self):
        # 1) yield the current node
        yield self
        # 2) recursively yield from children
        for child in self._children:
            yield from child


class Operator(Node):

    def is_same(self, other):
        if other.__class__ == self.__class__:
            return True
        return False

class Atom(Node):
    pass


class BinaryVariable(Atom):
    """Binary attribute, interpreted as x[:, IDX_ATTR] == 1."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        if isinstance(other, BinaryVariable):
            return self.name == other.name
        else:
            return False


class CategoricalVariable(Atom):
    """Categorical attribute, interpreted as x[:, IDX_ATTR] == value."""

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.name}={self.value}"

    def __eq__(self, other):
        if isinstance(other, CategoricalVariable):
            # TODO: this might not be the beast idea
            return self.name == other.name and self.value == other.value
        else:
            return False


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

        # Negation only applies to propositions (BinaryVariable or CategoricalVariable)
        if not isinstance(
            expr,
            (
                BinaryVariable,
                CategoricalVariable,
            ),
        ):
            raise ValueError(
                "Negation can only be applied to a variable-level proposition"
            )

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
        """
        attribute_index_map:
            maps attribute names to index constants (e.g., 'color' -> 'IDX_COLOR')
        category_value_map:
            maps attribute names to a dict of {human_readable_value: integer_value}.
            If an attribute is present here, it's considered categorical.
            If not present, we assume it's a binary attribute (0/1).
        """
        self.attribute_index_map = attribute_index_map
        self.category_value_map = (
            category_value_map if category_value_map is not None else {}
        )
        self.variables = list(attribute_index_map.keys())

        # Operators now include '=' for categorical checks
        self.operators = ["and", "or", "not", "->", "xor", "either", "(", ")", "="]

        # Maps internal variable identifiers (like "class_stop_sign", "color_0") to string names
        self.var_mappings = {}

    def tokenize(self, s):
        """
        Tokenize the input string into operators, variables, values, and parentheses.
        We'll ensure '=' is recognized as a separate token.
        """
        s = s.replace("->", " -> ")
        s = s.replace("(", " ( ")
        s = s.replace(")", " ) ")
        s = s.replace("=", " = ")

        tokens = s.lower().split()
        return tokens

    class Parser:
        def __init__(self, tokens, variables, operators, compiler):
            self.tokens = tokens
            self.pos = 0
            self.variables = variables
            self.operators = operators
            self.compiler = compiler  # to access category_value_map, etc.

        def peek_token(self):
            if self.pos < len(self.tokens):
                return self.tokens[self.pos]
            else:
                return None

        def consume_token(self, expected_token=None):
            if self.pos < len(self.tokens):
                token = self.tokens[self.pos]
                if expected_token is not None and token != expected_token:
                    raise ValueError(
                        f"Expected token '{expected_token}', got '{token}'"
                    )
                self.pos += 1
                return token
            else:
                return None

        #
        # Top-Level Parsing Methods
        #
        def parse_expression(self) -> Node:
            # 'either' is optional syntax we handle
            if self.peek_token() == "either":
                self.consume_token("either")
            node = self.parse_implies()
            return node

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
                if op == "or":
                    node = Or(node, right)
                elif op == "xor":
                    node = Xor(node, right)
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
            else:
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
            else:
                return self.parse_variable()

        #
        # Variable Parsing
        #
        def parse_variable(self):
            """
            A variable can be:
               - 'attr' (meaning attr=1 if it's binary)
               - 'attr=some_value' (either integer or category name)
            We also allow negation at the parser level: 'not attr', etc.
            """
            attr_token = self.consume_token()
            if attr_token not in self.variables:
                raise ValueError(f"Unknown attribute '{attr_token}'")

            # If user wrote "attr=value"...
            if self.peek_token() == "=":
                self.consume_token("=")  # consume '='
                value_token = self.consume_token()  # e.g., "stop_sign" or "1"
                value_int = self.resolve_value(attr_token, value_token)
                return CategoricalVariable(attr_token, value_int)
            else:
                # If no "=", check if it's effectively a binary attribute, or a "binary-like" category with 2 values.
                if self.is_binary(attr_token):
                    # 'A' alone => A=1
                    return BinaryVariable(attr_token)
                else:
                    # The attribute is truly multi-valued, so user must specify e.g. color=red
                    raise ValueError(
                        f"Attribute '{attr_token}' is multi-valued; please specify '{attr_token}=...'"
                    )

        def resolve_value(self, attr, value_token):
            """
            Convert 'value_token' into an integer code.
            If the attribute is in category_value_map, we do a lookup if 'value_token' is not a digit.
            Otherwise, for binary or unregistered attributes, we expect digits.
            """
            # If attribute is recognized as categorical
            if attr in self.compiler.category_value_map:
                cat_map = self.compiler.category_value_map[attr]
                # If user typed a digit
                if value_token.isdigit():
                    val = int(value_token)
                    # Optional: check if val is in cat_map.values()?
                    # We'll skip that for brevity
                    return val
                else:
                    # Look up the human-readable name
                    if value_token in cat_map:
                        return cat_map[value_token]
                    else:
                        raise ValueError(
                            f"Unknown category '{value_token}' for attribute '{attr}'"
                        )
            else:
                # Not in category_value_map => a "binary" attribute,
                # or user is specifying an integer code anyway
                if not value_token.isdigit():
                    raise ValueError(
                        f"Attribute '{attr}' is not categorical, please use a digit or shorthand 'attr' for =1."
                    )
                return int(value_token)

        def is_binary(self, attr):
            """
            Decide if 'attr' can be used in binary form (shorthand 'attr' => x[:,IDX_ATTR]==1).
            - If not in category_value_map, it's assumed binary.
            - If in category_value_map but has exactly 2 possible values, also allow 'attr' => x[:,IDX_ATTR]==1.
            - Otherwise, it's multi-valued and requires explicit =value.
            """
            if attr not in self.compiler.category_value_map:
                return True  # not in map => old-style binary
            else:
                # If cat_map has exactly 2 possible categories, we can interpret 'attr' as 'attr=1'
                cat_map = self.compiler.category_value_map[attr]
                return len(cat_map) == 2

    #
    # Code Generation
    #
    def get_var_name(self, var_name):
        """
        Generate a unique variable name for code generation.
        Example: "class_stop_sign", "color_red", or just "male".
        """
        if var_name not in self.var_mappings:
            code_var_name = var_name
            self.var_mappings[var_name] = code_var_name
        else:
            code_var_name = self.var_mappings[var_name]
        return code_var_name

    def collect_definitions(self):
        """
        # TODO: preliminary but working
        """
        definitions = []
        for var_map_key in self.var_mappings:
            # 1) If it's a known attribute name in attribute_index_map, treat as binary
            if var_map_key in self.attribute_index_map:
                idx_name = self.attribute_index_map[var_map_key]
                definitions.append(f"{var_map_key} = x[:, {idx_name}] == 1")

            # 2) Otherwise, if it has underscore, treat as categorical
            elif "_" in var_map_key:
                attr, val_str = var_map_key.rsplit("_", 1)
                # attempt to parse the integer
                val = int(val_str)
                idx_name = self.attribute_index_map[attr]
                definitions.append(f"{var_map_key} = x[:, {idx_name}] == {val}")

            # 3) Otherwise, raise error
            else:
                raise ValueError(
                    f"Could not interpret '{var_map_key}'. "
                    "It's neither a known binary attribute nor 'attr_value' format."
                )

        return definitions

    def generate_constraint_expression(self, node):
        """
        Recursively generate the Python boolean expression for the constraint.
        """
        if isinstance(node, BinaryVariable):
            # e.g. user wrote "A", meaning A=1 => we define "A" => x[:, IDX_A]==1
            var_name = self.get_var_name(node.name)
            return var_name

        elif isinstance(node, CategoricalVariable):
            # e.g. "class=stop_sign"
            var_name = self.get_var_name(f"{node.name}_{node.value}")
            return var_name

        elif isinstance(node, Not):
            expr = self.generate_constraint_expression(node.child)
            return f"~({expr})"

        elif isinstance(node, And):
            left = self.generate_constraint_expression(node.left)
            right = self.generate_constraint_expression(node.right)
            return f"({left}) & ({right})"

        elif isinstance(node, Or):
            left = self.generate_constraint_expression(node.left)
            right = self.generate_constraint_expression(node.right)
            return f"({left}) | ({right})"

        elif isinstance(node, Xor):
            left = self.generate_constraint_expression(node.left)
            right = self.generate_constraint_expression(node.right)
            return f"({left}) ^ ({right})"

        elif isinstance(node, Implies):
            # A -> B => (~A) | B
            left = self.generate_constraint_expression(node.left)
            right = self.generate_constraint_expression(node.right)
            return f"(~({left})) | ({right})"

        else:
            raise ValueError("Unknown node type")

    def generate_satisfaction_condition(self, node):
        """
        Return the code expression that checks if this constraint is satisfied.
        """
        return self.generate_constraint_expression(node)

    def compile(self, function_name, input_string):
        """
        Compile the constraint string (e.g. "class=stop_sign -> shape=octagon and color=red")
        into Python code that returns 1 if satisfied, 0 if not.
        """
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
        code_lines.append(
            f"    return torch.where({condition_expr}, 1, 0).unsqueeze(1)"
        )
        return "\n".join(code_lines)


#
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
    # print(And() in ast)
    #
    # print(Xor() in ast)
