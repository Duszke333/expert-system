import numpy as np


class Node:
    """
    An object that holds information about data split to smaller datasets,
    or a decision if it's a leaf node (no data can be split anymore).

    :param _variable: str; name of the rule variable by which data is split.

    :param _subnodes: dict; keys are possible answers to _variable
    (or '<=' and '>' if the rule variable values are numerical (threshold exists))
    values are Nodes made with smaller data that has the variable value corresponding
    to the key.

    :param _threshold: float; used only when _variable values are numerical,
    holds the threshold value of the data split to two smaller sets: with
    less or equal value than threshold and higher value,
    which are used to make Nodes with further splits (or decisions).

    :param _decision: type depending on the rule dataset; only used when dataset given
    to make the node cannot be split further (e.g. maximum depth has been reached)
    or all outcomes of the dataset are identical.
    Then a created node only holds the decision.
    """
    def __init__(self, variable=None, subnodes=None, threshold=None, decision=None):
        self._variable = variable
        self._subnodes = subnodes
        self._threshold = threshold

        # used for leaves only
        self._decision = decision

    @property
    def variable(self):
        """Getter for variable."""
        return self._variable

    @property
    def subnodes(self):
        """Getter for subnodes."""
        return self._subnodes

    @property
    def threshold(self):
        """Getter for threshold."""
        return self._threshold

    @property
    def decision(self):
        """Getter for decision."""
        return self._decision


class DecisionTree:
    """
    An object that creates a network of Nodes that form a decision tree.

    :param _outcome_header: str; name of the outcome variable.

    :param _variables: list; a list of non-outcome rule variable names.

    :param _data: NumPy array; segregated data where the outcome variable values are
    located in the last column, the result of 'prepare_data' method.

    :param _max_tree_depth: int; determines how many splits can a tree branch have,
    its value is the number of non-outcome rule variables.

    :param _root: Node object; defaults to None if self._data is empty,
    holds the core dataset splitting Node.
    Value is the result of 'build_the_tree' function.
    """
    def __init__(self, data, outcome_header):
        self._outcome_header = outcome_header
        variables = list(data.columns)
        variables.remove(outcome_header)
        self._variables = variables
        self._data = self.prepare_data(data)
        self._max_tree_depth = float('inf')
        # Uncommenting the line below will limit the number of questions
        # at the cost of probability of getting the correct answer
        # self._max_tree_depth = len(self.variables)
        self._root = self.build_the_tree(self.data) if np.any(self.data) else None

    @property
    def outcome_header(self):
        """Getter for outcome header"""
        return self._outcome_header

    @property
    def variables(self):
        """Getter for variables"""
        return self._variables

    @property
    def data(self):
        """Getter for data"""
        return self._data

    @property
    def max_tree_depth(self):
        """Getter for max tree depth"""
        return self._max_tree_depth

    @property
    def root(self):
        """Getter for root"""
        return self._root

    def prepare_data(self, data):
        """
        A method that changes given pandas DataFrame object to a NumPy array
        where outcome values are located in the last column, then returns it.
        """
        other_feature_values = data.drop(columns=self.outcome_header).values
        outcome_values = data[[self.outcome_header]].values
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

    def numerical_split(self, dataset, variable_index, previous_best_split):
        """
        A method that searches for the best numerical value to be the threshold
        of the data split, then checks if the found split is better
        than the one passed as the method argument.
        Returns a dictionary that consists of:
        -index of the variable that the data is split by,
        -a dictionary that consists of keys stating the relation of variable values in arrays
        of split data to the threshold (<= or >) and values being the split data arrays,
        -threshold value,
        -the information gain of the split,
        It will return the passed split if all the checked splits are worse than it.
        """
        best_gain = previous_best_split['info_gain']
        best_split = {}
        potential_thresholds = np.unique(dataset[:, variable_index])
        for threshold in potential_thresholds:
            less_or_equal_data = np.array([row for row in dataset if row[variable_index] <= threshold])
            higher_data = np.array([row for row in dataset if row[variable_index] > threshold])
            if len(less_or_equal_data) > 0 and len(higher_data) > 0:
                outcomes = dataset[:, -1]
                smaller_data_outcomes = less_or_equal_data[:, -1]
                bigger_data_outcomes = higher_data[:, -1]
                split_outcomes = [smaller_data_outcomes, bigger_data_outcomes]
                current_info_gain = self.information_gain(outcomes, split_outcomes)
                if current_info_gain > best_gain:
                    data_subsets = {
                        '<=': less_or_equal_data,
                        '>': higher_data
                    }
                    best_split = self.build_split(variable_index, data_subsets, threshold, current_info_gain)
                    best_gain = current_info_gain
        if not best_split:
            return previous_best_split
        return best_split

    def variable_split(self, dataset, variable_index, previous_best_split):
        """
        A method that splits data by a variable with non-numerical values
        and checks if the information gain of this split
        is better than the one passed as the method argument.
        Returns a dictionary that consists of:
        -index of the variable that the data is split by,
        -a dictionary that consists of keys being the variable values
        (e.g. for 'animal class' variable one of the keys would be 'Mammal')
        and values being the split data arrays,
        -the information gain of the split.
        It will return the passed split if all the checked splits are worse than it.
        """
        new_gain = previous_best_split['info_gain']
        best_split = {}
        variable_values = np.unique(dataset[:, variable_index])
        sliced_data_list = []
        sliced_outcome_list = []
        outcomes = dataset[:, -1]
        for value in variable_values:
            slice = np.array([row for row in dataset if row[variable_index] == value])
            sliced_data_list.append(slice)
            sliced_outcome_list.append(slice[:, -1])
        current_info_gain = self.information_gain(outcomes, sliced_outcome_list)
        if current_info_gain > new_gain:
            data_subsets = {}
            for index, value in enumerate(variable_values):
                data_subsets[value] = sliced_data_list[index]
            best_split = self.build_split(variable_index, data_subsets, None, current_info_gain)
        if not best_split:
            return previous_best_split
        return best_split

    def build_split(self, variable_index, data_subsets, threshold, gain):
        """
        A method that builds and returns a dictionary
        containing information about the data split.
        """
        split = {
            'variable_index': variable_index,
            'data_subsets': data_subsets,
            'threshold': threshold,
            'info_gain': gain
        }
        return split

    def determine_best_split(self, dataset):
        """
        A method that checks every variable in given dataset and finds the best split
        (calculated by calling numerical_split and variable_split functions) for it.
        Returns the 'best_split' dictionary which contains information about the split.
        """
        best_split = {'info_gain': -float('inf')}
        all_variables_values = dataset[:, :-1]
        for variable_index in range(len(self.variables)):
            variable_values = np.unique(all_variables_values[:, variable_index])
            num_types = (int, float, np.float64)
            if all(type(variable_value) in num_types for variable_value in variable_values):
                best_split = self.numerical_split(dataset, variable_index, best_split)
            else:
                best_split = self.variable_split(dataset, variable_index, best_split)
                pass
        return best_split

    def calculate_leaf_value(self, values):
        """
        A method that returns the most repeated value in given array of outcomes.
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
            best_possible_split = self.determine_best_split(dataset)
            if best_possible_split['info_gain'] > 0:
                subnodes = {}
                for variable_value, sliced_data in best_possible_split['data_subsets'].items():
                    subnodes[variable_value] = self.build_the_tree(sliced_data, current_depth + 1)
                variable_index = best_possible_split['variable_index']
                variable = self.variables[variable_index]
                return Node(variable, subnodes, best_possible_split['threshold'])
        leaf_value = self.calculate_leaf_value(dataset[:, -1])
        return Node(decision=leaf_value)

    def make_prediction(self, x, node=None):
        """
        A method that returns the outcome value matched for the given
        array of other variables' values.
        """
        if not node:
            node = self.root
        if node.decision is not None:
            return node.decision
        feature_value = x[self.variables.index(node.variable)]
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self.make_prediction(x, node.subnodes['<='])
            else:
                return self.make_prediction(x, node.subnodes['>'])
        else:
            return self.make_prediction(x, node.subnodes[feature_value])

    def accuracy(self):
        """
        A method that calculates the accuracy of the decision tree,
        which is the ratio of correct decisions made to the number of given rules.
        Returns a string informing about the result.
        """
        other_values = self.data[:, :-1]
        outcome_values = self.data[:, -1]
        predicted_values = [self.make_prediction(x) for x in other_values]
        correct_predictions = 0
        for index, y in enumerate(predicted_values):
            if y == outcome_values[index]:
                correct_predictions += 1
        accuracy_percentage = round(correct_predictions / len(predicted_values) * 100, 2)
        correct_decisions = f'{correct_predictions} / {len(outcome_values)}'
        return f'The answer accuracy is {accuracy_percentage}% ({correct_decisions})'

    def printer(self, node=None, indent='  ', answer=''):
        """
        A method that prints the decision tree.
        Used only for checking purposes, not shown in the actual program.
        """
        if not node:
            node = self.root
        if node.decision is not None:
            msg = indent + str(answer) + ' - ' + str(node.decision)
            print(msg)
        else:
            message = indent
            if answer:
                message += answer + ': '
            message += str(node.variable) + ': '
            if node.threshold:
                message += f'<= {node.threshold}'
            message += '?'
            print(message)
            for answer, subnode in node.subnodes.items():
                self.printer(subnode, indent + '  ', str(answer))
