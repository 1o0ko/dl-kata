'''
Core functions of tinyflow
'''
from collections import defaultdict


def build_graph(feed_dict):
    '''
    Builds graph from list on input nodes

    Args:
        in_nodes - dictionary with input nodes

    Returns:
        graph - sortable graph representation
    '''
    graph = defaultdict(lambda: {'in': set(), 'out': set()})
    nodes = [node for node in feed_dict]
    while nodes:
        node = nodes.pop(0)
        for out_node in node.outbound_nodes:
            graph[node]['out'].add(out_node)
            graph[out_node]['in'].add(node)
            nodes.append(out_node)

    return graph


def topological_sort(graph):
    """
    Sort the nodes in topological order.
    Returns a list of sorted nodes.
    """
    # NOTE: Input nodes have empty 'in' key
    L = []
    S = [node for node in graph if not graph[node]['in']]
    while S:
        n = S.pop(0)
        L.append(n)
        for m in n.outbound_nodes:
            graph[n]['out'].remove(m)
            graph[m]['in'].remove(n)
            # if no other incoming edges add to S
            if not graph[m]['in']:
                S.append(m)
    return L


def value_and_grad(node, feed_dict, wrt=None):
    """
    Performs a forward and backward pass. The value of `node` after the forward
    pass will be returned along with the gradients of all nodes in `wrt`.

    Arguments:
        node:      A node in the graph, should be the output node
                   (have no outgoing edges)
        feed_dict: A dictionary where the key is a `Input` node and the value
                   is the respective value feed to that node.
        wrt:       A list of nodes. The gradient for each node will be returned
    """
    assert node.outbound_nodes == []

    # use empy list if None
    wrt = wrt if wrt else []

    nodes = topological_sort(build_graph(feed_dict))

    # forward pass
    for n in nodes:
        n.forward(feed_dict)

    # backward pass
    for n in nodes[::-1]:
        n.backward(feed_dict)

    return node.value, [n.gradients[n] for n in wrt]
