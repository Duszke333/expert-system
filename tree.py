import numpy as np


class Node:
    """
    An object that holds information about data split to smaller datasets,
    or a value if it's a leaf node (no data can be split anymore).

    :param _feature: str; name of the feature by which data is split.

    :param _subnodes: dict; keys are possible answers to feature
    (or '<=' and '>' if the feature values are numerical (threshold exists))
    values are Nodes made with smaller data that has the feature value corresponding
    to the key.

    :param _threshold: float; used only when feature values are numerical,
    holds the threshold value of the data split to two smaller sets: with
    less or equal value than threshold and higher value,
    which are used to make Nodes with further splits (or values).

    :param _value: type depending on the problem; only used when dataset given
    to make the node cannot be split further (e.g. maximum depth has been reached)
    or all outcomes of the dataset are identical.
    Then a created node only holds the value.
    """
    def __init__(self, feature=None, subnodes=None, threshold=None, value=None):
        self._feature = feature
        self._subnodes = subnodes
        self._threshold = threshold
        # for leaves only
        self._value = value

    @property
    def feature(self):
        """Getter for feature."""
        return self._feature

    @property
    def subnodes(self):
        """Getter for subnodes."""
        return self._subnodes

    @property
    def threshold(self):
        """Getter for threshold."""
        return self._threshold

    @property
    def value(self):
        """Getter for value."""
        return self._value


class DecisionTree:
    """
    An object that creates a network of Nodes that form a decision tree.

    :param outcome_header: str; name of the outcome feature.

    :param features: list; a list of non-outcome feature names.

    :param data: NumPy array; segregated data where the outcome values are
    located in the last column, the result of 'prepare_data' method.

    :param max_tree_depth: int; determines how many splits can a tree brach have,
    its value is the number of non-outcome features.

    :param root: Node object; defaults to None if self.data is empty,
    holds the core dataset splitting Node.
    Value is the result of 'build_the_tree' function.
    """
    def __init__(self, data, outcome_header):
        self.outcome_header = outcome_header
        features = list(data.columns)
        features.remove(outcome_header)
        self.features = features
        self.data = self.prepare_data(data, self.outcome_header)
        self.max_tree_depth = len(self.features)
        # self.max_tree_depth = float('inf')
        self.root = self.build_the_tree(self.data) if np.any(self.data) else None

    def prepare_data(self, data, outcome_header):
        """
        A method that changes given pandas DataFrame object to a NumPy array
        where outcome values are located in the last column, then returns it.
        """
        other_feature_values = data.drop(columns=outcome_header).values
        outcome_values = data[[outcome_header]].values
        data = np.concatenate((other_feature_values, outcome_values), axis=1)
        return data

    def gini_index_calculator(self, outcome_values):
        """
        A method that calculates the Gini index of a dataset which is a probability
        of a specific feature being classified incorrectly when picked randomly.
        The mathematic formula is: 1 - sum of squared probabilities of values being picked.
        Returns a float number in <0; 1> range, where lower value is better.
        """
        unique_outcomes = np.unique(outcome_values)
        probabilities = 0
        for outcome in unique_outcomes:
            outcome_probability = len(outcome_values[outcome_values == outcome]) / len(outcome_values)
            probabilities += outcome_probability ** 2
        return 1 - probabilities

    def information_gain(self, data_outcomes, split_data_outcomes):
        """
        A method that calculates the improvment in data purity
        of a given split of given dataset.
        Returns a float number, where higher value is better.
        """
        old_gini = self.gini_index_calculator(data_outcomes)
        new_gini = 0
        for smaller_data_outcomes in split_data_outcomes:
            smaller_gini = self.gini_index_calculator(smaller_data_outcomes)
            new_gini += smaller_gini * len(smaller_data_outcomes) / len(data_outcomes)
        return old_gini - new_gini

    def numerical_split(self, dataset, feature_index, previous_best_split):
        """
        A method that searches for the best numerical value to be the threshold
        of the data split, then checks if the found split is better
        than the one passed as the method argument.
        Returns a dictionary that consists of:
        -the feature that of which values the data is split by,
        -a dictionary that consists of keys stating the relation of feature values in arrays
        of split data to the threshold (<= or >) and values being the split data arrays,
        -threshold value,
        -the information gain of the split,
        It will return the passed split if all the checked splits are worse than it.
        """
        new_gain = previous_best_split['info_gain']
        best_split = {}
        potential_thresholds = np.unique(dataset[:, feature_index])
        for threshold in potential_thresholds:
            less_or_equal_data = np.array([row for row in dataset if row[feature_index] <= threshold])
            higher_data = np.array([row for row in dataset if row[feature_index] > threshold])
            if len(less_or_equal_data) > 0 and len(higher_data) > 0:
                outcomes = dataset[:, -1]
                smaller_data_outcomes = less_or_equal_data[:, -1]
                bigger_data_outcomes = higher_data[:, -1]
                split_outcomes = [smaller_data_outcomes, bigger_data_outcomes]
                current_info_gain = self.information_gain(outcomes, split_outcomes)
                if current_info_gain > new_gain:
                    data_subsets = {
                        '<=': less_or_equal_data,
                        '>': higher_data
                    }
                    best_split = self.build_split(feature_index, data_subsets, threshold, current_info_gain)
                    new_gain = current_info_gain
        if not best_split:
            return previous_best_split
        return best_split

    def feature_split(self, dataset, feature_index, previous_best_split):
        """
        A method that splits data by a feature with non-numerical values
        and checks if the information gain of this split
        is better than the one passed as the method argument.
        Returns a dictionary that consists of:
        -the feature that the data is sorted by,
        -a dictionary that consists of keys being the feature values
        (e.g. for 'animal class' feature key would be 'Mammal')
        and values being the split data arrays,
        -the information gain of the split.
        It will return the passed split if all the checked splits are worse than it.
        """
        new_gain = previous_best_split['info_gain']
        best_split = {}
        feature_values = np.unique(dataset[:, feature_index])
        sliced_data_list = []
        sliced_outcome_list = []
        outcomes = dataset[:, -1]
        for value in feature_values:
            slice = np.array([row for row in dataset if row[feature_index] == value])
            sliced_data_list.append(slice)
            sliced_outcome_list.append(slice[:, -1])
        current_info_gain = self.information_gain(outcomes, sliced_outcome_list)
        if current_info_gain > new_gain:
            data_subsets = {}
            for index, value in enumerate(feature_values):
                data_subsets[value] = sliced_data_list[index]
            best_split = self.build_split(feature_index, data_subsets, None, current_info_gain)
        if not best_split:
            return previous_best_split
        return best_split

    def build_split(self, feature_index, data_subsets, threshold, gain):
        """
        A method that builds and returns a dictionary
        containing information about the data split.
        """
        split = {
            'feature': feature_index,
            'data_subsets': data_subsets,
            'threshold': threshold,
            'info_gain': gain
        }
        return split

    def get_best_split(self, dataset):
        """
        A method that checks every feature in given dataset and finds the best split
        (calculated by calling numerical_split and feature_split functions) for it.
        Returns the 'best_split' dictionary which contains information about the split.
        """
        best_split = {'info_gain': -float('inf')}
        features = dataset[:, :-1]
        for feature in range(len(self.features)):
            feature_values = np.unique(features[:, feature])
            num_types = (int, float, np.float64)
            if all(type(feature_value) in num_types for feature_value in feature_values):
                best_split = self.numerical_split(dataset, feature, best_split)
            else:
                best_split = self.feature_split(dataset, feature, best_split)
                pass
        return best_split

    def calculate_leaf_value(self, values):
        """
        A method that returns the most repeated value in given array of values.
        """
        values = list(values)
        return max(values, key=values.count)

    def build_the_tree(self, dataset, current_depth=1):
        """
        A funcion that creates the network of Nodes that form a decision tree
        by splitting the data recursively until the max depth has been reached
        or smaller dataset cannot be split further, then creates proper Nodes
        and connects them into the network.
        """
        is_data_pure = len(np.unique(dataset[:, -1])) == 1
        if current_depth <= self.max_tree_depth and not is_data_pure:
            best_possible_split = self.get_best_split(dataset)
            if best_possible_split['info_gain'] > 0:
                subnodes = {}
                for feature_value, sliced_data in best_possible_split['data_subsets'].items():
                    subnodes[feature_value] = self.build_the_tree(sliced_data, current_depth + 1)
                feature_index = best_possible_split['feature']
                feature = self.features[feature_index]
                return Node(feature, subnodes, best_possible_split['threshold'])
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        return Node(value=leaf_value)

    def coverage(self):
        """
        A method that calculates the coverage of the decision tree,
        which is the ratio of correct decision to the number of given rules.
        Returns a string informing about the result.
        """
        other_values = self.data[:, :-1]
        outcome_values = self.data[:, -1]
        predicted_values = [self.make_prediction(x) for x in other_values]
        correct_predictions = 0
        for index, y in enumerate(predicted_values):
            if y == outcome_values[index]:
                correct_predictions += 1
        coverage_percentage = round(correct_predictions / len(predicted_values) * 100, 2)
        correct_guesses = f'{correct_predictions} / {len(outcome_values)}'
        return f'The data coverage is {coverage_percentage}% ({correct_guesses})'

    def make_prediction(self, x, node=None):
        """
        A method that returns the outcome value matched for the given
        array of other features' values.
        """
        if not node:
            node = self.root
        if node.value is not None:
            return node.value
        feature_value = x[self.features.index(node.feature)]
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self.make_prediction(x, node.subnodes['<='])
            else:
                return self.make_prediction(x, node.subnodes['>'])
        else:
            return self.make_prediction(x, node.subnodes[feature_value])

    def printer(self, node=None, indent='  ', answer=''):
        """
        A method that prints the decision tree.
        Used only for checking purposes, not shown in the actual program.
        """
        if not node:
            node = self.root
        if node.value is not None:
            msg = indent + str(answer) + ' - ' + str(node.value)
            print(msg)
        else:
            message = indent
            if answer:
                message += answer + ': '
            message += str(node.feature) + ': '
            if node.threshold:
                message += f'<= {node.threshold}'
            message += '?'
            print(message)
            for answer, subnode in node.subnodes.items():
                self.printer(subnode, indent + '  ', str(answer))
