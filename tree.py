import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, feature=None, subnodes=None, threshold=None, info_gain=None, value=None):
        self.feature = feature
        self.subnodes = subnodes
        self.threshold = threshold
        self.info_gain = None
        # for leaves only
        self.value = value


class TreeCreator:
    def __init__(self, max_tree_depth=2, min_samples_split=2):
        self.root = None
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split

    def gini_index_calculator(self, target_data):
        target_labels = np.unique(target_data)
        gini = 0
        for target in target_labels:
            target_probability = len(target_data[target_data == target]) / len(target_data)
            gini += target_probability ** 2
        return 1 - gini

    def information_gain(self, parent_data, split_data_list):
        old_gini = self.gini_index_calculator(parent_data)
        new_gini = 0
        for child_data in split_data_list:
            new_gini += self.gini_index_calculator(child_data) * len(child_data) / len(parent_data)
        return old_gini - new_gini

    def numerical_split(self, dataset, attribute, current_best_split, max_gain):
        best_split_num = {}
        potential_thresholds = np.unique(dataset[:, attribute])
        for threshold in potential_thresholds:
            less_or_equal_data = np.array([row for row in dataset if row[attribute] <= threshold])
            higher_data = np.array([row for row in dataset if row[attribute] > threshold])
            if len(less_or_equal_data) > 0 and len(higher_data) > 0:
                # targets = dataset[[target]]
                targets = dataset[:, -1]
                smaller_targets = less_or_equal_data[:, -1]
                bigger_targets = higher_data[:, -1]
                current_info_gain = self.information_gain(targets, [smaller_targets, bigger_targets])
                if current_info_gain > max_gain:
                    data_subsets = {
                        'Less': less_or_equal_data,
                        'More': higher_data
                    }
                    best_split_num = self.build_split(attribute, data_subsets, threshold, current_info_gain)
                    max_gain = current_info_gain
        if not best_split_num:
            return current_best_split
        return best_split_num

    def attribute_split(self, dataset, attribute, target, current_best_split, max_gain):
        best_split = {}
        attribute_values = np.unique(dataset[:, attribute])
        sliced_data_list = []
        sliced_target_list = []
        targets = dataset[[target]]
        for value in attribute_values:
            slice = np.array([row for row in dataset if row[attribute] == value])
            sliced_data_list.append(slice)
            sliced_target_list.append(slice[[target]])
        current_info_gain = self.information_gain(targets, sliced_target_list)
        if current_info_gain > max_gain:
            data_subsets = {}
            for index, value in enumerate(attribute_values):
                data_subsets[value] = sliced_data_list[index]
            best_split = self.build_split(attribute, data_subsets, None, current_info_gain)
        if not best_split:
            return current_best_split
        return best_split

    def build_split(self, attribute, data_subsets, threshold, gain):
        split = {
            'feature': attribute,
            'data_subsets': data_subsets,
            'threshold': threshold,
            'info_gain': gain
        }
        return split

    def get_best_split(self, dataset, target, num_features):
        best_split = {}
        max_gain = -float('inf')
        attributes, targets = self.split_data(dataset, target)
        # attribute_names = attributes.columns
        # for attribute in attribute_names:
        for attribute in range(num_features):
            attribute_values = np.unique(attributes[:, attribute])
            if type(attribute_values[0]) in (int, float, np.float64):
                best_split = self.numerical_split(dataset, attribute, attribute_values, max_gain)
            else:
                best_split = self.attribute_split(dataset, target, max_gain)
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

    def build_the_tree(self, dataset, target_label, current_depth=1):
        num_samples, num_features = np.shape(dataset)
        if num_samples >= self.min_samples_split and current_depth <= self.max_tree_depth:
            num_features = np.shape(dataset)[1] - 1
            best_possible_split = self.get_best_split(dataset, target_label, num_features)
            if best_possible_split['info_gain'] > 0:
                subnodes = {}
                for attribute_value, sliced_data in best_possible_split['data_subsets'].items():
                    subnodes[attribute_value] = self.build_the_tree(sliced_data, target_label, current_depth + 1)
                return Node(best_possible_split['feature'], subnodes, best_possible_split['threshold'])
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        return Node(value=leaf_value)

    def fit(self, X, Y, target):
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_the_tree(data, target)

    def predictions(self, X):
        predicitons = [self.make_prediction(x, self.root) for x in X]
        return predicitons

    def make_prediction(self, x, node):
        if node.value is not None:
            return node.value
        feature_value = x[node.feature]
        if type(feature_value) in (int, float, np.float64):
            if feature_value <= node.threshold:
                return self.make_prediction(x, node.subnodes['Less'])
            else:
                return self.make_prediction(x, node.subnodes['More'])
        else:
            return self.make_prediction(x, node.subnodes[feature_value])

    def printer(self, node=None, indent=' '):
        if not node:
            node = self.root
        if node.value is not None:
            print(node.value)
        else:
            message = str(node.feature)
            if node.threshold:
                message += str(node.threshold)
            message += f'? {node.info_gain}'
            print(message)
            for subnode in node.subnodes.values():
                self.printer(subnode, indent + ' ')


def main():
    data = pd.read_csv('./drzewo decyzyjne/datasets/iris.csv')
    attributes = data.columns
    target_label = 'Type'
    X = data.drop(columns=target_label).values
    Y = data[[target_label]].values.reshape(-1, 1)
    tree = TreeCreator(3, 2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    tree.fit(X_train, Y_train, target_label)
    tree.printer()
    Y_pred = tree.predictions(X_test)
    print(accuracy_score(Y_test, Y_pred))
    pass


if __name__ == '__main__':
    main()
