import pandas as pd


class Node:
    def __init__(self, feature=None, subnodes=None, threshold=None, value=None):
        self.feature = feature
        self.subnodes = subnodes
        self.threshold = threshold
        # for leaves only
        self.value = value


def main():
    data = pd.read_csv('./drzewo decyzyjne/datasets/iris.csv')
    attributes = data.columns
    target_label = 'Type'
    print(attributes[1])
    x = data.drop(columns=target_label)
    y = data[[target_label]]
    pass


if __name__ == '__main__':
    main()
