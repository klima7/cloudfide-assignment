import pandas as pd
from solution import add_virtual_column


def assert_empty_frame(df: pd.DataFrame) -> None:
    assert df.empty and df.columns.empty


def test_rejects_non_dataframe_input():
    result = add_virtual_column("not a df", "first_column", "new_column")
    assert_empty_frame(result)


def test_rejects_non_string_role_or_column():
    frame = pd.DataFrame({"first_column": [1]})
    role_result = add_virtual_column(frame, 123, "new_column")
    column_result = add_virtual_column(frame, "first_column", 456)
    assert_empty_frame(role_result)
    assert_empty_frame(column_result)


def test_rejects_invalid_existing_column_labels():
    frame = pd.DataFrame([[1, 2]], columns=["valid", "invalid-name"])
    result = add_virtual_column(frame, "valid + invalid_name", "new_column")
    assert_empty_frame(result)


def test_rejects_invalid_new_column_name():
    frame = pd.DataFrame({"valid": [1, 2]})
    result = add_virtual_column(frame, "valid + 1", "invalid name")
    assert_empty_frame(result)


def test_rejects_duplicate_new_column_name():
    frame = pd.DataFrame({"valid": [1, 2]})
    result = add_virtual_column(frame, "valid + 1", "valid")
    assert_empty_frame(result)


def test_rejects_dependency_on_new_column():
    frame = pd.DataFrame({"first_column": [1, 2]})
    result = add_virtual_column(frame, "first_column + new_column", "new_column")
    assert_empty_frame(result)


def test_rejects_unknown_column_dependency():
    frame = pd.DataFrame({"first_column": [1, 2]})
    result = add_virtual_column(frame, "first_column + missing_column", "new_column")
    assert_empty_frame(result)


def test_rejects_unsupported_operations():
    frame = pd.DataFrame({"first_column": [2, 4], "second_column": [1, 2]})
    result = add_virtual_column(frame, "first_column / second_column", "new_column")
    assert_empty_frame(result)


def test_rejects_invalid_syntax():
    frame = pd.DataFrame({"first_column": [1]})
    result = add_virtual_column(frame, "first_column +", "new_column")
    assert_empty_frame(result)


def test_supports_unary_operations_and_constants():
    frame = pd.DataFrame({"first_column": [1, -2, 3]})
    result = add_virtual_column(frame, "-first_column + 3", "computed")
    expected = pd.DataFrame(
        {"first_column": [1, -2, 3], "computed": [2, 5, 0]},
        dtype="int64",
    )
    pd.testing.assert_frame_equal(result, expected)


def test_supports_constant_only_expression():
    frame = pd.DataFrame({"first_column": [10, 20]})
    result = add_virtual_column(frame, "3", "constant_column")
    expected = pd.DataFrame(
        {"first_column": [10, 20], "constant_column": [3, 3]},
        dtype="int64",
    )
    pd.testing.assert_frame_equal(result, expected)


def test_returns_empty_when_evaluation_fails():
    frame = pd.DataFrame({"first_column": ["a", "b"]})
    result = add_virtual_column(frame, "first_column - first_column", "computed")
    assert_empty_frame(result)


def test_complex_expression_across_multiple_columns():
    frame = pd.DataFrame(
        {
            "col_a": [2, 3],
            "col_b": [5, 7],
            "col_c": [11, 13],
            "col_d": [17, 19],
        }
    )
    result = add_virtual_column(
        frame,
        "col_a * col_b + col_c - col_d",
        "computed",
    )
    expected = pd.DataFrame(
        {
            "col_a": [2, 3],
            "col_b": [5, 7],
            "col_c": [11, 13],
            "col_d": [17, 19],
            "computed": [2 * 5 + 11 - 17, 3 * 7 + 13 - 19],
        },
        dtype="int64",
    )
    pd.testing.assert_frame_equal(result, expected)


def test_does_not_mutate_original_dataframe():
    frame = pd.DataFrame({"first_column": [1, 2]})
    _ = add_virtual_column(frame, "first_column + 1", "computed")
    assert "computed" not in frame.columns
