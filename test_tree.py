import numpy as np
import pandas as pd
from tree import Node, DecisionTree
from tree import validate_choice, validate_yes_no


def test_init_node_empty():
    node = Node()
    assert node._feature is None
    assert node._subnodes is None
    assert node._threshold is None
    assert node._value is None


def test_init_node_numerical():
    node1 = Node()
    node2 = Node()
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)
    assert node._feature == 'sample'
    assert node._subnodes['<='] == node1
    assert node._subnodes['>'] == node2
    assert node._threshold == 2.5
    assert node._value is None


def test_init_node_not_numerical():
    node1 = Node()
    node2 = Node()
    subnodes = {
        'A': node1,
        'B': node2
    }
    node = Node('letter', subnodes)
    assert node._feature == 'letter'
    assert node._subnodes['A'] == node1
    assert node._subnodes['B'] == node2
    assert node._threshold is None
    assert node._value is None


def test_init_leaf_node():
    node = Node(value='Dog')
    assert node._feature is None
    assert node._subnodes is None
    assert node._threshold is None
    assert node._value == 'Dog'


def test_guess_leaf_node():
    node = Node(value='Dog')
    assert node.make_a_decision() == 'Dog'


def test_guess_numerical_node_smaller_value(monkeypatch):
    node1 = Node(value=2)
    node2 = Node(value=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  Yes    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert node.make_a_decision() == 2


def test_guess_numerical_node_equal_value(monkeypatch):
    node1 = Node(value=2.5)
    node2 = Node(value=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  Yes    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert node.make_a_decision() == 2.5


def test_guess_numerical_node_higher_value(monkeypatch):
    node1 = Node(value=2)
    node2 = Node(value=3)
    subnodes = {
        '<=': node1,
        '>': node2
    }
    node = Node('sample', subnodes, 2.5)

    def mon_input(_):
        return '  nO    '
    monkeypatch.setattr('builtins.input', mon_input)
    assert node.make_a_decision() == 3


def test_guess_not_numerical_node(monkeypatch):
    node1 = Node(value='a')
    node2 = Node(value='b')
    subnodes = {
        'A': node1,
        'B': node2
    }
    node = Node('letter', subnodes)

    def mon_input(_):
        return 'A'
    monkeypatch.setattr('builtins.input', mon_input)
    assert node.make_a_decision() == 'a'


def test_init_tree_empty():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    assert tree.target_name == 'Doll'
    assert tree.features == ['Dog', 'Bike']
    assert not tree.data
    assert tree.max_tree_depth == 2
    assert tree.root is None


def test_prepare_data():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Dog'],
        ['Yes', 4, 'Dog']],
        dtype=object
    )
    df = pd.DataFrame(data, columns=['Decision', 'Age', 'Animal'])
    expected_result = np.array([
        [3, 'Dog', 'Yes'],
        [5, 'Dog', 'No'],
        [4, 'Dog', 'Yes']],
        dtype=object
    )
    assert np.array_equal(tree.prepare_data(df, 'Decision'), expected_result)


def test_tree_gini_index_calculator():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data = np.array(['Bike', 'Dog', 'Dog', 'Doll', 'Dog'])
    expected = 1 - (1/5)**2 - (3/5)**2 - (1/5)**2
    assert tree.gini_index_calculator(data) == expected


def test_tree_information_gain():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data = np.array(['Bike', 'Dog', 'Dog', 'Doll', 'Dog'])
    data1 = np.array(['Dog', 'Dog', 'Dog'])
    data2 = np.array(['Bike', 'Doll'])
    data_expected = 1 - (1/5)**2 - (3/5)**2 - (1/5)**2
    assert tree.gini_index_calculator(data) == data_expected
    data1_expected = 1 - (3/3)**2
    assert tree.gini_index_calculator(data1) == data1_expected
    data2_expected = 1 - (1/2)**2 - (1/2) ** 2
    assert tree.gini_index_calculator(data2) == data2_expected
    info_gain_expected = data_expected - data1_expected * (3/5) - data2_expected * (2/5)
    assert tree.information_gain(data, [data1, data2]) == info_gain_expected


def test_tree_numerical_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    split = {'info_gain': -float('inf')}
    data_to_split = np.array([
        [False, 2, 'Bike'],
        [True, 4, 'Dog'],
        [True, 2.5, 'Bike'],
        [False, 3, 'Bike'],
        [True, 3.5, 'Dog']
    ])
    expected_smaller = np.array([
        [False, 2, 'Bike'],
        [True, 2.5, 'Bike'],
        [False, 3, 'Bike']
    ])
    expected_bigger = np.array([
        [True, 4, 'Dog'],
        [True, 3.5, 'Dog']
    ])
    expected_gain = tree.information_gain(data_to_split[:, -1],
                                          [expected_smaller[:, -1],
                                           expected_bigger[:, -1]])
    new_split = tree.numerical_split(data_to_split, 1, split)
    assert new_split['feature'] == 1
    assert np.array_equal(new_split['data_subsets']['<='], expected_smaller)
    assert np.array_equal(new_split['data_subsets']['>'], expected_bigger)
    assert new_split['threshold'] == '3'
    assert new_split['info_gain'] == expected_gain


def test_tree_numerical_split_no_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data_to_split = np.array([
        [False, 2, 'Dog'],
        [True, 4, 'Dog'],
        [True, 2.5, 'Dog'],
        [False, 3, 'Dog'],
        [True, 3.5, 'Dog']
    ])
    split = {'info_gain': 1}
    new_split = tree.numerical_split(data_to_split, 1, split)
    assert new_split == split


def test_tree_feature_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data_to_split = np.array([
        ['Yes', 2, 'Bike'],
        ['No', 4, 'Dog'],
        ['Yes', 2.5, 'Bike'],
        ['Yes', 3, 'Doll'],
        ['No', 3.5, 'Car']
    ])
    expected_1 = np.array([
        ['Yes', 2, 'Bike'],
        ['Yes', 2.5, 'Bike'],
        ['Yes', 3, 'Doll']
    ])
    expected_2 = np.array([
        ['No', 4, 'Dog'],
        ['No', 3.5, 'Car']
    ])
    expected_gain = tree.information_gain(data_to_split[:, -1], [expected_1[:, -1], expected_2[:, -1]])
    split = {'info_gain': -float('inf')}
    new_split = tree.feature_split(data_to_split, 0, split)
    assert new_split['feature'] == 0
    assert np.array_equal(new_split['data_subsets']['Yes'], expected_1)
    assert np.array_equal(new_split['data_subsets']['No'], expected_2)
    assert new_split['threshold'] is None
    assert new_split['info_gain'] == expected_gain


def test_tree_feature_split_no_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data_to_split = np.array([
        ['Yes', 2, 'Bike'],
        ['No', 4, 'Dog'],
        ['Yes', 2.5, 'Bike'],
        ['Yes', 3, 'Doll'],
        ['No', 3.5, 'Car']
    ])
    split = {'info_gain': 1}
    new_split = tree.feature_split(data_to_split, 0, split)
    assert new_split == split


def test_build_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    subsets = {
        'A': np.array(['A', 2.5, True]),
        'B': np.array(['B', 3, False])
    }
    split = tree.build_split(0, subsets, None, 1)
    assert split['feature'] == 0
    assert split['data_subsets'] == subsets
    assert split['threshold'] is None
    assert split['info_gain'] == 1


def test_get_best_split_numerical_over_feature():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data_to_split = np.array([
        ['Yes', 3.5, 'Dog'],
        ['No', 3, 'Bike'],
        ['No', 5, 'Dog'],
        ['Yes', 4, 'Dog'],
        ['No', 1.5, 'Bike']],
        dtype=object
    )
    expected_smaller = np.array([
        ['No', 3, 'Bike'],
        ['No', 1.5, 'Bike']],
        dtype=object
    )
    expected_bigger = np.array([
        ['Yes', 3.5, 'Dog'],
        ['No', 5, 'Dog'],
        ['Yes', 4, 'Dog']],
        dtype=object
    )
    expected_gain = tree.information_gain(data_to_split[:, -1],
                                          [expected_smaller[:, -1],
                                           expected_bigger[:, -1]])
    split = tree.get_best_split(data_to_split, 2)
    assert split['feature'] == 1
    assert np.array_equal(split['data_subsets']['<='], expected_smaller)
    assert np.array_equal(split['data_subsets']['>'], expected_bigger)
    assert split['threshold'] == 3
    assert split['info_gain'] == expected_gain


def test_get_best_split_feature_over_numerical():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    data_to_split = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 3, 'Bike'],
        ['Yes', 5, 'Dog'],
        ['Yes', 5, 'Dog'],
        ['No', 3, 'Bike']],
        dtype=object
    )
    expected_yes = np.array([
        ['Yes', 3, 'Dog'],
        ['Yes', 5, 'Dog'],
        ['Yes', 5, 'Dog']],
        dtype=object
    )
    expected_no = np.array([
        ['No', 3, 'Bike'],
        ['No', 3, 'Bike']],
        dtype=object
    )
    expected_gain = tree.information_gain(data_to_split[:, -1],
                                          [expected_yes[:, -1],
                                           expected_no[:, -1]])
    split = tree.get_best_split(data_to_split, 2)
    assert split['feature'] == 0
    assert np.array_equal(split['data_subsets']['Yes'], expected_yes)
    assert np.array_equal(split['data_subsets']['No'], expected_no)
    assert split['threshold'] is None
    assert split['info_gain'] == expected_gain


def test_tree_calculate_leaf_value_simple():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    values = np.array([
        'Bike', 'Bike', 'Bike', 'Bike'],
        dtype=object
    )
    value = tree.calculate_leaf_value(values)
    assert value == 'Bike'


def test_tree_calculate_leaf_value_different_values():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    values = np.array([
        'Dog', 'Bike', 'Dog', 'Bike', 'Doll'],
        dtype=object
    )
    value = tree.calculate_leaf_value(values)
    assert value == 'Dog'


def test_tree_build_the_tree_simple():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    tree.features = ['Decision', 'Age']
    tree.max_tree_depth = 2
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Dog'],
        ['Yes', 4, 'Dog']],
        dtype=object
    )
    root = tree.build_the_tree(data)
    assert root.feature is None
    assert root.subnodes is None
    assert root.threshold is None
    assert root.value == 'Dog'


def test_tree_build_the_tree_with_feature_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    tree.features = ['Decision', 'Age']
    tree.max_tree_depth = 2
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['Yes', 4, 'Dog']],
        dtype=object
    )
    root = tree.build_the_tree(data)
    assert root.feature == 'Decision'
    assert root.subnodes is not None
    assert root.threshold is None
    assert root.value is None
    yes_node = root.subnodes['Yes']
    assert yes_node.feature is None
    assert yes_node.subnodes is None
    assert yes_node.threshold is None
    assert yes_node.value == 'Dog'
    no_node = root.subnodes['No']
    assert no_node.feature is None
    assert no_node.subnodes is None
    assert no_node.threshold is None
    assert no_node.value == 'Cat'


def test_tree_build_the_tree_numerical_split():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    tree.features = ['Decision', 'Age']
    tree.max_tree_depth = 2
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog']],
        dtype=object
    )
    root = tree.build_the_tree(data)
    assert root.feature == 'Age'
    assert root.subnodes is not None
    assert root.threshold == 4
    assert root.value is None
    smaller_node = root.subnodes['<=']
    assert smaller_node.feature is None
    assert smaller_node.subnodes is None
    assert smaller_node.threshold is None
    assert smaller_node.value == 'Dog'
    bigger_node = root.subnodes['>']
    assert bigger_node.feature is None
    assert bigger_node.subnodes is None
    assert bigger_node.threshold is None
    assert bigger_node.value == 'Cat'


def test_tree_build_the_tree_not_enough_depth():
    sample_features = ['Dog', 'Bike', 'Doll']
    sample_data = pd.DataFrame(columns=sample_features)
    tree = DecisionTree(sample_data, 'Doll')
    tree.features = ['Decision', 'Age']
    tree.max_tree_depth = 1
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    root = tree.build_the_tree(data)
    assert root.feature == 'Age'
    assert root.subnodes is not None
    assert root.threshold == 4
    assert root.value is None
    smaller_node = root.subnodes['<=']
    assert smaller_node.feature is None
    assert smaller_node.subnodes is None
    assert smaller_node.threshold is None
    assert smaller_node.value == 'Dog'
    bigger_node = root.subnodes['>']
    # Turtle and Cat cannot be split because the maximum depth has been passed
    assert bigger_node.feature is None
    assert bigger_node.subnodes is None
    assert bigger_node.threshold is None
    assert bigger_node.value == 'Cat'


def test_tree_begin_guessing():
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Dog'],
        ['Yes', 4, 'Dog']],
        dtype=object
    )
    df = pd.DataFrame(data, columns=['Decision', 'Age', 'Animal'])
    tree = DecisionTree(df, 'Animal')
    assert tree.decide() == 'Dog'


def test_tree_coverage():
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    df = pd.DataFrame(data, columns=['Decision', 'Age', 'Animal'])
    tree = DecisionTree(df, 'Animal')
    assert tree.coverage() == 'The data coverage is 100.0% (4 / 4)'


def test_tree_make_prediction():
    data = np.array([
        ['Yes', 3, 'Dog'],
        ['No', 5, 'Cat'],
        ['No', 4, 'Dog'],
        ['Yes', 7, 'Turtle']],
        dtype=object
    )
    df = pd.DataFrame(data, columns=['Decision', 'Age', 'Animal'])
    tree = DecisionTree(df, 'Animal')
    assert tree.make_prediction(np.array(['Yes', 3], dtype=object)) == 'Dog'
    assert tree.make_prediction(np.array(['No', 5], dtype=object)) == 'Cat'
    assert tree.make_prediction(np.array(['No', 4], dtype=object)) == 'Dog'
    assert tree.make_prediction(np.array(['Yes', 7], dtype=object)) == 'Turtle'


def test_validate_choice_already_valid():
    first_choice = 'Bike'
    choices = {
        'Bike': True,
        'Dog': 4,
        'Doll': None
    }
    assert validate_choice(first_choice, choices) == first_choice


def test_validate_choice_invalid_first(monkeypatch):
    inputs = iter(['Bob', 'Cat', '    Dog '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    first_choice = 'Monk'
    choices = {
        'Bike': True,
        'Dog': 4,
        'Doll': None
    }
    assert validate_choice(first_choice, choices) == 'Dog'


def test_validate_yes_no_already_valid():
    assert validate_yes_no('YeS') is True
    assert validate_yes_no('nO') is False


def test_validate_yes_no_first_invalid_no(monkeypatch):
    inputs = iter(['Bob', 'Cat', '    No '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert validate_yes_no('Bike') is False


def test_validate_yes_no_first_invalid_yes(monkeypatch):
    inputs = iter(['Bob', 'Cat', '    yEs '])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert validate_yes_no('Bike') is True
