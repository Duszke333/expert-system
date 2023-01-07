import numpy as np
from data_io import read_data


class Node:
    """
    An object that holds information about data split to smaller datasets,
    or a value if it's a leaf node (no data can be split anymore)

    :param feature: str; name of the feature by which data is split

    :param subnodes: dict; keys are possible answers to feature
    (or 'Less' and 'More' if the feature values are numerical (threshold exists))
    values are Nodes made with smaller data that has the feature value corresponding
    to the key

    :param threshold: float; used only when feature values are numerical,
    the value of the best split of dataset which is divided to two smaller sets: with
    less or equal value than threshold and higher value,
    which are used to make Nodes with further splits (or values)

    :param value: type depending on the problem; only used when dataset given
    to make the node cannot be split further or all target feature values of
    the dataset are equal. Then a created node only holds the value
    """
    def __init__(self, feature=None, subnodes=None, threshold=None, value=None):
        self.feature = feature
        self.subnodes = subnodes
        self.threshold = threshold
        # for leaves only
        self.value = value

    def guess(self):
        """
        An interactive method that asks the user questions, finds the searched
        target feature value recursively based on his answers and returns it.
        """
        if self.value is not None:
            return self.value
        if self.threshold:
            print(f'Does the "{self.feature}" feature value satisfy ', end='')
            print(f'the inequality: <={self.threshold} [Yes/No]?')
            choice = validate_yes_no(input('>>').strip())
            if choice is True:
                return self.subnodes['<='].guess()
            else:
                return self.subnodes['>'].guess()
        else:
            possible_answers = list(self.subnodes.keys())
            print('Please choose the value that matches the ', end='')
            print(f'"{self.feature}" feature of your data object from the list below:')
            print(possible_answers)
            choice = validate_choice(input('>>').strip(), self.subnodes)
            return self.subnodes[choice].guess()


class DecisionTree:
    """
    An object that creates a network of Nodes that form a decision tree

    :param root: Node object, defaults to None, Node is assigned by self.fit() method;
    It holds the first dataset splitting Node

    :param max_tree_depth: int; determines how many split nodes can a tree branch have
    """
    def __init__(self, max_tree_depth=2):
        self.root = None
        self.max_tree_depth = max_tree_depth

    def gini_index_calculator(self, target_data):
        """
        A method that calculates the Gini index of a dataset
        which is a probability of a specific feature being classified incorrectly
        when picked randomly.
        The mathematic formula is: 1 - sum of squared probabilities of values being picked
        It returns a float number in <0; 1> range, where lower value is better
        """
        target_labels = np.unique(target_data)
        gini = 0
        for target in target_labels:
            target_probability = len(target_data[target_data == target]) / len(target_data)
            gini += target_probability ** 2
        return 1 - gini

    def information_gain(self, parent_data, split_data_list):
        """
        A method that calculates the improvment in data purity
        of a given split of given dataset
        Returns a float number, where higher value is better
        """
        old_gini = self.gini_index_calculator(parent_data)
        new_gini = 0
        for child_data in split_data_list:
            new_gini += self.gini_index_calculator(child_data) * len(child_data) / len(parent_data)
        return old_gini - new_gini

    def numerical_split(self, dataset, feature_index, previous_best_split):
        """
        A method that searches for the best numerical value to be the threshold
        of the data split and check if it's better
        than the one passed as a method argument.
        Returns a dictionary that consists of:
        -the feature that the data is sorted by,
        -a dictionary that consists of keys stating the relation
        of feature values in arrays of split data to the threshold and the values
        -threshold value
        -the information gain of the split
        It will return the passed split if all the checked splits are worse than it
        """
        new_gain = previous_best_split['info_gain']
        best_split = {}
        potential_thresholds = np.unique(dataset[:, feature_index])
        for threshold in potential_thresholds:
            less_or_equal_data = np.array([row for row in dataset if row[feature_index] <= threshold])
            higher_data = np.array([row for row in dataset if row[feature_index] > threshold])
            if len(less_or_equal_data) > 0 and len(higher_data) > 0:
                targets = dataset[:, -1]
                smaller_targets = less_or_equal_data[:, -1]
                bigger_targets = higher_data[:, -1]
                current_info_gain = self.information_gain(targets, [smaller_targets, bigger_targets])
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
        -a dictionary that consists of keys stating the relation
        of feature values in arrays of split data to the threshold and the values
        -the information gain of the split
        It will return the passed split if all the checked splits are worse than it
        """
        new_gain = previous_best_split['info_gain']
        best_split = {}
        feature_values = np.unique(dataset[:, feature_index])
        sliced_data_list = []
        sliced_target_list = []
        targets = dataset[:, -1]
        for value in feature_values:
            slice = np.array([row for row in dataset if row[feature_index] == value])
            sliced_data_list.append(slice)
            sliced_target_list.append(slice[:, -1])
        current_info_gain = self.information_gain(targets, sliced_target_list)
        if current_info_gain > new_gain:
            data_subsets = {}
            for index, value in enumerate(feature_values):
                data_subsets[value] = sliced_data_list[index]
            best_split = self.build_split(feature_index, data_subsets, None, current_info_gain)
        if not best_split:
            return previous_best_split
        return best_split

    def build_split(self, feature, data_subsets, threshold, gain):
        """
        A method that builds and returns a dictionary
        containing information about the data split
        """
        split = {
            'feature': feature,
            'data_subsets': data_subsets,
            'threshold': threshold,
            'info_gain': gain
        }
        return split

    def get_best_split(self, dataset, num_features):
        """
        A method that checks every feature in given dataset and finds the best split
        (calculated by calling numerical_split and feature_split functions)
        for it, then returns it
        """
        best_split = {'info_gain': -float('inf')}
        features = dataset[:, :-1]
        for feature in range(num_features):
            feature_values = np.unique(features[:, feature])
            if all(type(feat_val) in (int, float, np.float64) for feat_val in feature_values):
                best_split = self.numerical_split(dataset, feature, best_split)
            else:
                best_split = self.feature_split(dataset, feature, best_split)
                pass
        return best_split

    def calculate_leaf_value(self, values):
        """
        A method that returns the most repeated value in given array of values
        """
        values = list(values)
        return max(values, key=values.count)

    def build_the_tree(self, dataset, features, current_depth=1):
        """
        A funcion that creates the network of Nodes that form a decision tree
        by splitting the data recursively until the max depth has been reached
        or smaller dataset cannot be split further, then creates proper Nodes
        and connects them into the network
        """
        if current_depth <= self.max_tree_depth:
            num_features = np.shape(dataset)[1] - 1
            best_possible_split = self.get_best_split(dataset, num_features)
            if best_possible_split['info_gain'] > 0:
                subnodes = {}
                for feature_value, sliced_data in best_possible_split['data_subsets'].items():
                    subnodes[feature_value] = self.build_the_tree(sliced_data, features, current_depth + 1)
                feat_index = best_possible_split['feature']
                feat = features[feat_index]
                return Node(feat, subnodes, best_possible_split['threshold'])
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        return Node(value=leaf_value)

    def fit(self, X, Y, features):
        """
        A method that puts the target feature values collumn on the end of dataset
        and then sets the tree root to the Node network by calling build_the_tree method
        """
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_the_tree(data, features)

    def coverage(self, X, features):
        """
        A method that returns a list of results of matching
        the target feature value to value arrays of a given dataset
        """
        coverage = [self.make_prediction(x, self.root, features) for x in X]
        return coverage

    def make_prediction(self, x, node, features):
        """
        A method that return the target feature value matched for the given
        array of feature values
        """
        if node.value is not None:
            return node.value
        feature_value = x[features.index(node.feature)]
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self.make_prediction(x, node.subnodes['<='], features)
            else:
                return self.make_prediction(x, node.subnodes['>'], features)
        else:
            return self.make_prediction(x, node.subnodes[feature_value], features)

    def printer(self, node=None, indent='  ', answer=''):
        """
        A method that prints the decision tree.
        Used only for checking purposes, not shown in the actual program
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

    def begin_guessing(self):
        """
        A method that starts the guessing of the target feature value
        and returns it
        """
        return self.root.guess()


def coverage_test(X, Y, tree, features):
    """
    A function that returns the float value of the ratio of correct
    predictions to the correct ones i. e. Checks the coverage
    of the tree for data in the source file
    and returns the information about it
    """
    predictions = tree.coverage(X, features)
    correct_predictions = 0
    for index, y in enumerate(predictions):
        if y == Y[index]:
            correct_predictions += 1
    coverage_percentage = round(correct_predictions / len(predictions) * 100, 2)
    correct_guesses = f'{correct_predictions} / {len(Y)}'
    return f'The data coverage is {coverage_percentage}% ({correct_guesses})'


def separate_data(data, target_feature):
    """
    A function that separates target feature values from other values
    and returns them along with the list of other features' names
    """
    features = list(data.columns)
    features.remove(target_feature)
    X = data.drop(columns=target_feature).values
    Y = data[[target_feature]].values.reshape(-1, 1)
    return X, Y, features


def validate_choice(choice, possible_choices):
    while choice not in possible_choices.keys():
        choice = input('Unrecognized choice. Please choose again: ').strip()
    return choice


def validate_yes_no(choice):
    while choice.lower() not in ['yes', 'no']:
        choice = input('Unrecognized choice. Please choose again [Yes/No]: ').strip()
    return True if choice.lower() == 'yes' else False


def main():
    """
    Used for ongoing testing
    """
    data = read_data('./drzewo decyzyjne/datasets/iris.csv')
    X, Y, features = separate_data(data, 'Type')
    tree = DecisionTree(len(features))
    tree.fit(X, Y, features)
    tree.printer()
    print(coverage_test(X, Y, tree, features))
    print(tree.begin_guessing())
    pass


if __name__ == '__main__':
    main()
