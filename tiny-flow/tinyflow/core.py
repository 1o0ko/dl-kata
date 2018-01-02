'''
Core functions of tinyflow
'''
from ops import (
    ForwardNode,
    BackwardNode
)

def topological_sort(input_nodes):
    """
    Sort the nodes in topological order.
    All nodes should be reachable through the `input_nodes`.
    Returns a list of sorted nodes.
    """
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
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

    input_nodes = [n for n in feed_dict]
    nodes = topological_sort(input_nodes)

    # forward pass
    for n in nodes:
        if isinstance(n, ForwardNode):
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    # backward pass
    for n in nodes[::-1]:
        if isinstance(n, BackwardNode):
            g = feed_dict[n]
            n.backward(g)
        else:
            n.backward()

    return node.value, [n.gradients[n] for n in wrt]
