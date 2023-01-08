import main as mn
import pandas as pd
import numpy as np


# The only tested functions are add_new_rule_to_data()
# and input_from_keyboard() as every remaining function in that file
# relies on functions from different files that have already been tested
# (answer validation, reading data, writing data, creating the decision tree)


def test_add_new_rule_to_data(monkeypatch):
    inputs = iter(['   Yes ', '   10 ', '    Rabbit '])
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
    assert all(mn.add_new_rule_to_data(data_frame) == expected_data_frame) is True


def test_input_from_keyboard_with_data_already(monkeypatch):
    inputs = iter(['Yes', '10', 'Rabbit', 'YeS', 'No', '0.5', 'Horse', 'nO', 'nO'])
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
    res_data_frame, path = mn.input_data_from_keyboard(data_frame, expected_path)
    assert path == expected_path
    assert all(res_data_frame == expected_data_frame) is True


def test_input_from_keyboard_no_data(monkeypatch):
    inputs = iter(['Decision', 'Age', 'Animal', 'quit', 'Yes', '3', 'Dog', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age', 'Animal']
    expected_data = np.array([['Yes', 3, 'Dog']], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path = mn.input_data_from_keyboard()
    assert all(data == expected_data_frame) is True
    assert path is None


def test_input_from_keyboard_quit_before_2_features(monkeypatch):
    inputs = iter(['Decision', 'quit', 'Age', 'quit', 'Yes', '3', 'Dog', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age']
    expected_data = np.array([['Yes', 3]], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path = mn.input_data_from_keyboard()
    assert all(data == expected_data_frame) is True
    assert path is None


def test_input_from_keyboard_repeating_features(monkeypatch):
    inputs = iter(['Decision', 'Age', 'Animal', 'Age', 'quit', 'Yes', '3', 'Dog', 'nO', 'nO'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    expected_cols = ['Decision', 'Age', 'Animal']
    expected_data = np.array([['Yes', 3, 'Dog']], dtype=object)
    expected_data_frame = pd.DataFrame(expected_data, columns=expected_cols)
    data, path = mn.input_data_from_keyboard()
    assert all(data == expected_data_frame) is True
    assert path is None
