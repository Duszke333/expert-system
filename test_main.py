import pandas as pd
import numpy as np
from main import add_new_rule_to_data, input_data_from_keyboard, fill_dataset_with_rules


# The only tested functions are add_new_rule_to_data()
# and input_from_keyboard() as every remaining function in that file
# relies on functions from different files that have already been tested
# (answer validation, reading data, writing data, creating the decision tree)


def test_add_new_rule_to_data_no_repeat(monkeypatch):
    inputs = iter(['   Yes ', '   10 ', '    Rabbit ', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 10, 'Rabbit']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    add_new_rule_to_data(data_frame, 'Animal')
    assert all(data_frame.eq(expected_data_frame)) is True


def test_add_new_rule_to_data_empty_value_input(monkeypatch):
    inputs = iter(['Yes', '        ', '' '10', 'Rabbit', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 10, 'Rabbit']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    add_new_rule_to_data(data_frame, 'Animal')
    assert all(data_frame.eq(expected_data_frame)) is True


def test_add_new_rule_to_data_identical_data(monkeypatch):
    inputs = iter(['Yes', '3', 'Dog', 'Yes', '10', 'Rabbit', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 10, 'Rabbit']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    add_new_rule_to_data(data_frame, 'Animal')
    assert all(data_frame.eq(expected_data_frame)) is True


def test_add_new_rule_to_data_different_targets_no_replace(monkeypatch):
    inputs = iter(['Yes', '3', 'Duck', 'No', 'Yes', '10', 'Rabbit', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 10, 'Rabbit']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    add_new_rule_to_data(data_frame, 'Animal')
    assert all(data_frame.eq(expected_data_frame)) is True


def test_add_new_rule_to_data_different_targets_replace(monkeypatch):
    inputs = iter(['Yes', '3', 'Duck', 'Yes', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 3, 'Duck']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    add_new_rule_to_data(data_frame, 'Animal')
    assert all(data_frame.eq(expected_data_frame)) is True


def test_fill_dataset_with_rules(monkeypatch):
    inputs = iter(['Yes', '10', 'Rabbit', '', 'No', '0.5', 'Horse', '', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cols = ['Decision', 'Age', 'Animal']
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    data_frame = pd.DataFrame(data, columns=cols)
    expected_data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle'],
        ['Yes', 10, 'Rabbit'],
        ['No', 0.5, 'Horse']],
        dtype=object
    )
    expected_data_frame = pd.DataFrame(expected_data, columns=cols)
    expected_path = 'sample'
    res_data_frame, path, target = fill_dataset_with_rules(data_frame, expected_path, 'Animal')
    assert path == expected_path
    assert all(res_data_frame.eq(expected_data_frame)) is True
    assert target == 'Animal'


def test_input_from_keyboard_no_data(monkeypatch):
    inputs = iter(['Decision', 'Age', 'Animal', 'quit', 'Animal', 'Yes', '3', 'Dog', '', 'Yes', '5', 'Duck', '', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age', 'Animal']
    expected_data = np.array([['Yes', 3, 'Dog'], ['Yes', 5, 'Duck']], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path, target = input_data_from_keyboard()
    assert all(data.eq(expected_data_frame)) is True
    assert path is None
    assert target == 'Animal'


def test_input_from_keyboard_quit_before_2_features(monkeypatch):
    inputs = iter(['Decision', 'quit', 'Age', 'quit', 'Age', 'Yes', '3', 'Dog', '5', 'Duck', '', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age']
    expected_data = np.array([['Yes', 3], [5, 'Duck']], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path, target = input_data_from_keyboard()
    assert all(data.eq(expected_data_frame)) is True
    assert path is None
    assert target == 'Age'


def test_input_from_keyboard_repeating_features(monkeypatch):
    inputs = iter(['Decision', 'Age', 'Animal', 'Age', 'quit', 'Animal', 'Yes', '3', 'Dog', '', 'Yes', '5', 'Duck', '', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age', 'Animal']
    expected_data = np.array([['Yes', 3, 'Dog'], ['Yes', 5, 'Duck']], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path, target = input_data_from_keyboard()
    assert all(data.eq(expected_data_frame)) is True
    assert path is None
    assert target == 'Animal'
