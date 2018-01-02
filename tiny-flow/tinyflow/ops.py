'''
Brain-dead simple version of tensorflow-like DL framework based on graphs
'''
import numpy as np


class BaseNode(object):
    def __init__(self, inbound_nodes):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        # store here the values computed in the forward pass
        # that will be used in the backward pass
        self.cache = {}

        # set this value on the forward pass
        self.value = None
        # set these gradients on the backward pass
        # the key should be an input node and the
        # value the gradient for that node
        self.gradients = {}
        self.typname = type(self).__name__

        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)


class Node(BaseNode):
    ''' Specifies the API contract '''
    def __init__(self, inbound_nodes):
        BaseNode.__init__(self, inbound_nodes)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplementedError

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current node with respect
        to the input nodes. The gradient of the loss with respect
        to the current node should already be computed in the `gradients`
        attribute of the output nodes.
        """
        raise NotImplementedError


class BackwardNode(BaseNode):
    def __init__(self, inbound_nodes):
        BaseNode.__init__(self, inbound_nodes)

    def forward(self):
        raise NotImplementedError

    def backward(self, value):
        raise NotImplementedError

class ForwardNode(BaseNode):
    def __init__(self, inbound_nodes):
        BaseNode.__init__(self, inbound_nodes)

    def forward(self, value):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MockGrad(BackwardNode):
    ''' Used in tests '''
    def __init__(self, x):
        BackwardNode.__init__(self, [x])

    def forward(self):
        self.value = self.inbound_nodes[0].value

    def backward(self, value):
        self.gradients = {n: value for n in self.inbound_nodes}


class Input(ForwardNode):
    ''' Implements inputing values to the graph `'''
    def __init__(self):
        ForwardNode.__init__(self, [])

    # NOTE: Input node is the only node where the value is
    # passed as an argument to forward()
    #
    # All other node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    def forward(self, value):
        self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]


class Add(Node):
    ''' Implements binary addition: x + y '''
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = [node.value for node in self.inbound_nodes]
        self.value = x + y

    def backward(self):
        self.gradients = {n: 0 for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad = n.gradients[self]

            # NOTE: this is jus to make the operation explict
            x, y = self.inbound_nodes
            self.gradients[x] += 1 * grad
            self.gradients[y] += 1 * grad


class Mul(Node):
    ''' Implements binary multiplication: x * y '''
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = [node.value for node in self.inbound_nodes]
        self.value = x * y

    def backward(self):
        self.gradients = {n: 0 for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad = n.gradients[self]

            x, y = self.inbound_nodes
            self.gradients[x] += y.value * grad
            self.gradients[y] += x.value * grad


if __name__ == '__main__':
    print('yo!')
