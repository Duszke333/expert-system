import pandas as pd
import numpy as np


class Node:
    """
    An object that holds information about data split to smaller datasets,
    or a value if it's a leaf node (no data can be split anymore)

    :param feature: str; name of the attribute by which data is split

    :param subnodes: dict; keys are possible answers to feature
    (or 'Less' and 'More' if the feature values are numerical (threshold exists))
    values are Nodes made with smaller data that has the feature value corresponding
    to the key

    :param threshold: float; used only when feature values are numerical,
    the value of the best split of dataset which is divided to two smaller sets: with
    less or equal value than threshold and higher value,
    which are used to make Nodes with further splits (or values)

    :param value: type depending on the problem; only used when dataset given
    to make the node cannot be split further or all target attribute values of
    the dataset are equal. Then a created node only holds the value
    """
    def __init__(self, feature=None, subnodes=None, threshold=None, value=None):
        self.feature = feature
        self.subnodes = subnodes
        self.threshold = threshold
        # for leaves only
        self.value = value


class DecisionTree:
    """
    An object that creates a network of Nodes that form a decision tree

    :param root: Node object, defaults to None, Node is assigned by self.fit() function;
    It holds the first dataset splitting Node

    :param max_tree_depth: int; determines how many split nodes can a tree branch have
    """
    def __init__(self, max_tree_depth=2):
        self.root = None
        self.max_tree_depth = max_tree_depth

    def gini_index_calculator(self, target_data):
        """
        A function that calculates the Gini index of a dataset
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
        A function that calculates the improvment in data purity
        of a given split of given dataset
        Returns a float number, where higher value is better
        """
        old_gini = self.gini_index_calculator(parent_data)
        new_gini = 0
        for child_data in split_data_list:
            new_gini += self.gini_index_calculator(child_data) * len(child_data) / len(parent_data)
        return old_gini - new_gini

    def numerical_split(self, dataset, attribute, current_best_split, max_gain):
        """
        A function that searches for the best numerical value to be the threshold
        of the data split.
        Returns:
        a dictionary that consists of:
        -the feature that the data is sorted by,
        -a dictionary that consists of keys stating the relation
        of feature values in arrays of split data to the threshold and the values
        -threshold value
        -the information gain of the split
        and the
        """
        new_gain = max_gain
        best_split_num = {}
        potential_thresholds = np.unique(dataset[:, attribute])
        for threshold in potential_thresholds:
            less_or_equal_data = np.array([row for row in dataset if row[attribute] <= threshold])
            higher_data = np.array([row for row in dataset if row[attribute] > threshold])
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
                    best_split_num = self.build_split(attribute, data_subsets, threshold, current_info_gain)
                    new_gain = current_info_gain
        if not best_split_num:
            return current_best_split, new_gain
        return best_split_num, new_gain

    def attribute_split(self, dataset, attribute, target, current_best_split, max_gain):
        new_gain = max_gain
        best_split = {}
        attribute_values = np.unique(dataset[:, attribute])
        sliced_data_list = []
        sliced_target_list = []
        targets = dataset[:, -1]
        for value in attribute_values:
            slice = np.array([row for row in dataset if row[attribute] == value])
            sliced_data_list.append(slice)
            sliced_target_list.append(slice[:, -1])
        current_info_gain = self.information_gain(targets, sliced_target_list)
        if current_info_gain > new_gain:
            data_subsets = {}
            for index, value in enumerate(attribute_values):
                data_subsets[value] = sliced_data_list[index]
            best_split = self.build_split(attribute, data_subsets, None, current_info_gain)
            new_gain = current_info_gain
        if not best_split:
            return current_best_split, new_gain
        return best_split, new_gain

    def build_split(self, attribute, data_subsets, threshold, gain):
        split = {
            'feature': attribute,
            'data_subsets': data_subsets,
            'threshold': threshold,
            'info_gain': gain
        }
        return split

    def get_best_split(self, dataset, target, num_features):
        best_split = {'info_gain': -float('inf')}
        max_gain = -float('inf')
        attributes, _ = self.split_data(dataset, target)
        for attribute in range(num_features):
            attribute_values = np.unique(attributes[:, attribute])
            # if len(attribute_values) == 1:
            #     continue
            if all(type(attr_value) in (int, float, np.float64) for attr_value in attribute_values) and len(attribute_values) > 2:
                best_split, max_gain = self.numerical_split(dataset, attribute, best_split, max_gain)
            else:
                best_split, max_gain = self.attribute_split(dataset, attribute, target, best_split, max_gain)
                pass
        return best_split

    def split_data(self, dataset, target):
        # attributes = dataset.drop(columns=target)
        # targets = dataset[[target]]
        attributes = dataset[:, :-1]
        targets = dataset[:, -1]
        return attributes, targets

    def calculate_leaf_value(self, values):
        values = list(values)
        return max(values, key=values.count)

    def build_the_tree(self, dataset, target_label, attributes, current_depth=1):
        if current_depth <= self.max_tree_depth:
        # if num_samples >= self.min_samples_split and current_depth <= self.max_tree_depth:
            num_features = np.shape(dataset)[1] - 1
            best_possible_split = self.get_best_split(dataset, target_label, num_features)
            if best_possible_split['info_gain'] > 0:
                subnodes = {}
                for attribute_value, sliced_data in best_possible_split['data_subsets'].items():
                    subnodes[attribute_value] = self.build_the_tree(sliced_data, target_label, attributes, current_depth + 1)
                attr_index = best_possible_split['feature']
                attr = attributes[attr_index]
                return Node(attr, subnodes, best_possible_split['threshold'])
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        return Node(value=leaf_value)

    def fit(self, X, Y, target, attributes):
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_the_tree(data, target, attributes)

    def predictions(self, X, attributes):
        predicitons = [self.make_prediction(x, self.root, attributes) for x in X]
        return predicitons

    def make_prediction(self, x, node, attributes):
        if node.value is not None:
            return node.value
        feature_value = x[attributes.index(node.feature)]
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self.make_prediction(x, node.subnodes['<='], attributes)
            else:
                return self.make_prediction(x, node.subnodes['>'], attributes)
        else:
            return self.make_prediction(x, node.subnodes[feature_value], attributes)

    def printer(self, node=None, indent='  ', answer=''):
        if not node:
            node = self.root
        if node.value is not None:
            msg = indent + str(answer) + '-' + str(node.value)
            print(msg)
        else:
            message = indent
            if answer:
                message += answer + ': '
            message += str(node.feature) + ': '
            if node.threshold:
                message += f'<={node.threshold}'
            message += '?'
            print(message)
            for answer, subnode in node.subnodes.items():
                self.printer(subnode, indent + '  ', str(answer))


def coverage_test(X, Y, tree, attributes):
    predictions = tree.predictions(X, attributes)
    correct_predictions = 0
    for index, y in enumerate(predictions):
        if y == Y[index]:
            correct_predictions += 1
    return correct_predictions / len(predictions)


def main():
    data = pd.read_csv('./drzewo decyzyjne/datasets/iris.csv')
    attributes = list(data.columns)
    target_label = 'Type'
    attributes.remove(target_label)
    X = data.drop(columns=target_label).values
    Y = data[[target_label]].values.reshape(-1, 1)
    # depth = len(np.unique(data[[target_label]]))
    # tree = DecisionTree(depth)
    tree = DecisionTree(len(attributes))
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    # tree.fit(X_train, Y_train, target_label, attributes)
    tree.fit(X, Y, target_label, attributes)
    tree.printer()
    # Y_pred = tree.predictions(X_test, attributes)
    # print(accuracy_score(Y_test, Y_pred))
    print(coverage_test(X, Y, tree, attributes))
    pass


if __name__ == '__main__':
    main()
