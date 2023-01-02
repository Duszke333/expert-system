from tree import Node


def test_init_node_empty():
    node = Node()
    assert node.feature is None
    assert node.subnodes is None
    assert node.threshold is None
    assert node.value is None


def test_init_node_numerical():
    node1 = Node()
    node2 = Node()
    subnodes = {
        'Less': node1,
        'More': node2
    }
    node = Node('sample', subnodes, 2.5)
    assert node.feature == 'sample'
    assert list(node.subnodes.values())[0] == node1
    assert list(node.subnodes.values())[1] == node2
    assert node.threshold == 2.5
    assert node.value is None


def test_init_node_not_numerical():
    node1 = Node()
    node2 = Node()
    subnodes = {
        'A': node1,
        'B': node2
    }
    node = Node('letter', subnodes)
    assert node.feature == 'letter'
    assert list(node.subnodes.values())[0] == node1
    assert list(node.subnodes.values())[1] == node2
    assert node.threshold is None
    assert node.value is None


def test_init_leaf_node():
    node = Node(value='Dog')
    assert node.feature is None
    assert node.subnodes is None
    assert node.threshold is None
    assert node.value == 'Dog'
