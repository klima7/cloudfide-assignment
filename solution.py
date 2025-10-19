import ast
import re
from typing import Any, Iterable

import pandas as pd


COLUMN_NAME_PATTERN = re.compile(r"^[A-Za-z_]+$")


def _column_name_is_valid(column: Any) -> bool:
    """Return True if the value is a valid column identifier"""
    return isinstance(column, str) and COLUMN_NAME_PATTERN.fullmatch(column) is not None


class _ExpressionValidator(ast.NodeVisitor):
    """Validate supported syntax and collect column dependencies for role expression"""

    ALLOWED_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult)
    ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

    def __init__(self) -> None:
        self.column_names: set[str] = set()

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BINARY_OPS):
            raise ValueError("Unsupported operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.ALLOWED_UNARY_OPS):
            raise ValueError("Unsupported operator")
        self.visit(node.operand)

    def visit_Name(self, node: ast.Name) -> None:
        if not _column_name_is_valid(node.id):
            raise ValueError("Invalid column name")
        self.column_names.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")

    def visit_Num(self, node: ast.Num) -> None:  # pragma: no cover
        if not isinstance(node.n, (int, float)):
            raise ValueError("Only numeric constants are allowed")

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


def add_virtual_column(
    df: pd.DataFrame,
    role: str,
    new_column: str,
) -> pd.DataFrame:
    """Return a dataframe copy with `new_column` evaluated from `role`

    Args:
        df: Source dataframe containing columns used by the expression
        role: Arithmetic expression referencing existing column names
        new_column: Name for the computed column to create in the output

    Returns:
        A new dataframe with `new_column` populated, or an empty dataframe
        when validation or evaluation fails

    """
    if not _inputs_are_valid(df, role, new_column):
        return _empty_frame()

    try:
        tree, dependencies = _parse_role(role)
    except ValueError:
        return _empty_frame()

    if not _dependencies_are_valid(dependencies, df.columns, new_column):
        return _empty_frame()

    try:
        evaluated = _evaluate_expression(tree, df, dependencies)
    except ValueError:
        return _empty_frame()

    result_df = df.copy(deep=True)
    result_df[new_column] = evaluated.values
    return result_df


def _inputs_are_valid(
    df: Any,
    role: Any,
    new_column: Any,
) -> bool:
    """Validate the top-level inputs before processing"""
    if not isinstance(df, pd.DataFrame):
        return False
    if not isinstance(role, str) or not isinstance(new_column, str):
        return False
    if not all(_column_name_is_valid(column) for column in df.columns):
        return False
    if not _column_name_is_valid(new_column):
        return False
    if new_column in df.columns:
        return False
    return True


def _parse_role(role: str) -> tuple[ast.AST, set[str]]:
    """Parse the role expression

    Returns:
        A tuple containing the parsed AST and a set of dependency column names
    """
    expression = role.strip()
    if not expression:
        raise ValueError("Expression cannot be empty.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    validator = _ExpressionValidator()
    try:
        validator.visit(tree)
    except ValueError as exc:
        raise ValueError("Expression contains unsupported elements.") from exc
    return tree, set(validator.column_names)


def _dependencies_are_valid(
    dependencies: set[str],
    columns: Iterable[Any],
    new_column: str,
) -> bool:
    """Check that dependencies are available and do not reference the target"""
    if new_column in dependencies:
        return False
    column_set = set(columns)
    return dependencies.issubset(column_set)


def _evaluate_expression(
    tree: ast.AST,
    df: pd.DataFrame,
    dependencies: set[str],
) -> pd.Series:
    """Compile and evaluate the expression against the dataframe"""
    compiled = compile(tree, "<virtual_column>", "eval")
    env = {col: df[col] for col in dependencies}
    try:
        evaluated = eval(compiled, {"__builtins__": {}}, env)
    except Exception as exc:
        raise ValueError("Failed to evaluate expression.") from exc

    if not isinstance(evaluated, pd.Series):
        evaluated = pd.Series(evaluated, index=df.index)

    if len(evaluated) != len(df):
        raise ValueError("Evaluated result length does not match dataframe.")

    return evaluated


def _empty_frame() -> pd.DataFrame:
    """Return empty dataframe"""
    return pd.DataFrame([])
