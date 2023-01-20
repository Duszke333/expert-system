import pandas as pd
import numpy as np
from main import validate_choice, validate_yes_no
from main import add_new_rule_to_data, input_data_from_keyboard, fill_dataset_with_rules
from main import determine_decision
from tree import Node


# The only tested functions are add_new_rule_to_data()
# and input_from_keyboard() as every remaining function in that file
# relies on functions from different files that have already been tested
# (answer validation, reading data, writing data, creating the decision tree)


def test_validate_choice_already_valid(monkeypatch):
    def mon_input(_):
        return 'Bike'
    choices = {
        'Bike': True,
        'Dog': 4,
        'Doll': None
    }
    monkeypatch.setattr('builtins.input', mon_input)
    assert validate_choice(choices) == 'Bike'


def test_validate_choice_invalid_first(monkeypatch):
    inputs = iter(['Monk', 'Bob', 'Cat', '    Dog '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    choices = {
        'Bike': True,
        'Dog': 4,
        'Doll': None
    }
    assert validate_choice(choices) == 'Dog'


def test_validate_yes_no_already_yes(monkeypatch):
    def mon_input(_):
        return 'YeS'
    monkeypatch.setattr('builtins.input', mon_input)
    assert validate_yes_no() is True


def test_validate_yes_no_already_no(monkeypatch):
    def mon_input(_):
        return 'nO'
    monkeypatch.setattr('builtins.input', mon_input)
    assert validate_yes_no() is False


def test_validate_yes_no_first_invalid_no(monkeypatch):
    inputs = iter(['Bike', 'Bob', 'Cat', '    No '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert validate_yes_no() is False


def test_validate_yes_no_first_invalid_yes(monkeypatch):
    inputs = iter(['Bike', 'Bob', 'Cat', '    yEs '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert validate_yes_no() is True



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


def test_determine_decision_leaf_node():
    node = Node(decision='Dog')
    assert determine_decision(node) == 'Dog'


def test_determine_decision_numerical_node_smaller_value(monkeypatch):
    node1 = Node(decision=2)
    node2 = Node(decision=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  Yes    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 2


def test_determine_decision_numerical_node_equal_value(monkeypatch):
    node1 = Node(decision=2.5)
    node2 = Node(decision=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  Yes    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 2.5


def test_determine_decision_numerical_node_higher_value(monkeypatch):
    node1 = Node(decision=2)
    node2 = Node(decision=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  nO    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 3


def test_determine_decision_not_numerical_node(monkeypatch):
    node1 = Node(decision='a')
    node2 = Node(decision='b')
    subnodes = {
        'A': node1,
        'B': node2
    }
    node = Node('letter', subnodes)

    def mon_input(_):
        return 'A'
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 'a'


def test_determine_decision_bool_true(monkeypatch):
    node1 = Node(decision='a')
    node2 = Node(decision='b')
    subnodes = {
        True: node1,
        False: node2
    }
    node = Node('letter', subnodes)

    def mon_input(_):
        return '  yEs '
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 'a'


def test_determine_decision_bool_false(monkeypatch):
    node1 = Node(decision='a')
    node2 = Node(decision='b')
    subnodes = {
        True: node1,
        False: node2
    }
    node = Node('letter', subnodes)

    def mon_input(_):
        return '  nO '
    monkeypatch.setattr('builtins.input', mon_input)
    assert determine_decision(node) == 'b'
