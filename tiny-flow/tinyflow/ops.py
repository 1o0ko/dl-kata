'''
Brain-dead simple version of tensorflow-like DL framework based on graphs
'''
import numpy as np

from .math import softmax, sigmoid


class Node(object):
    ''' Specifies the API contract '''

    def __init__(self, inbound_nodes, name):
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

    # These will be implemented in a subclass.
    def forward(self, kvargs):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplementedError

    def backward(self, kvargs):
        """
        Backward propagation.

        Compute the gradient of the current node with respect
        to the input nodes. The gradient of the loss with respect
        to the current node should already be computed in the `gradients`
        attribute of the output nodes.
        """
        raise NotImplementedError


class MockGrad(Node):
    ''' Used in tests '''

    def __init__(self, x, name='MockGrad'):
        Node.__init__(self, [x], name)

    def forward(self, kvargs):
        self.value = self.inbound_nodes[0].value

    def backward(self, kvargs):
        grad = kvargs[self]
        self.gradients = {n: grad for n in self.inbound_nodes}


class Input(Node):
    ''' Implements inputing values to the graph `'''

    def __init__(self, name='InputNode'):
        Node.__init__(self, [], name)

    # NOTE: Input node is the only node where the value is
    # passed as an argument to forward()
    #
    # All other node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    def forward(self, kvargs):
        self.value = kvargs[self]

    def backward(self, kvargs):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]


class Add(Node):
    ''' Implements binary addition: x + y '''

    def __init__(self, x, y, name='AddNode'):
        Node.__init__(self, [x, y], name)

    def forward(self, kvargs):
        x, y = [node.value for node in self.inbound_nodes]
        self.value = x + y

    def backward(self, kvargs):
        self.gradients = {n: 0 for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad = n.gradients[self]

            # NOTE: this is jus to make the operation explict
            x, y = self.inbound_nodes
            self.gradients[x] += 1 * grad
            self.gradients[y] += 1 * grad


class Mul(Node):
    ''' Implements binary multiplication: x * y '''

    def __init__(self, x, y, name='Mul'):
        Node.__init__(self, [x, y], name)

    def forward(self, kvargs):
        x, y = [node.value for node in self.inbound_nodes]
        self.value = x * y

    def backward(self, kvargs):
        self.gradients = {n: 0 for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad = n.gradients[self]

            x, y = self.inbound_nodes
            self.gradients[x] += y.value * grad
            self.gradients[y] += x.value * grad


class Linear(Node):
    ''' Implements Linear layer: X*W + b '''

    def __init__(self, x_in, w_in, b_in, name):
        Node.__init__(self, [x_in, w_in, b_in], name)

    def forward(self, kvargs):
        X, W, b = [node.value for node in self.inbound_nodes]
        self.value = X @ W + b

    def backward(self, kvargs):
        '''
        For gradient calculations check:
            http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-gradient-notes.pdf
        '''
        self.gradients = {
            n: np.zeros_like(n.value) for n in self.inbound_nodes
        }

        for n in self.outbound_nodes:
            grad = n.gradients[self]

            X, W, b = self.inbound_nodes
            self.gradients[X] += grad @ W.value.T
            self.gradients[W] += X.value.T @ grad
            self.gradients[b] += np.sum(grad, axis=0, keepdims=False)


class Sigmoid(Node):
    '''
    Implements sigmoid activation function
    '''

    def __init__(self, x, name='Sigmoid'):
        Node.__init__(self, [x], name)

    def forward(self, kvargs):
        self.value = sigmoid(self.inbound_nodes[0].value)

    def backward(self, kvargs):
        self.gradients = {
            n: np.zeros_like(n.value) for n in self.inbound_nodes
        }
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            x = self.inbound_nodes[0]
            self.gradients[x] += grad_cost * self.value * (1 - self.value)


class Relu(Node):
    '''
    Implements ReLu activation function
    '''

    def __init__(self, x, name='Relu'):
        Node.__init__(self, [x], name=name)

    def forward(self, kvargs):
        x = self.inbound_nodes[0].value
        mask = x > 0
        self.value = x * mask

    def backward(self, kvargs):
        self.gradients = {
            n: np.zeros_like(n.value) for n in self.inbound_nodes
        }
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            x = self.inbound_nodes[0]
            self.gradients[x] += grad_cost * (self.value > 0)


class CrossEntropyWithLogits(Node):
    '''
    Implements categorical cross-entropy loss with softmax
    '''

    def __init__(self, x, y, name='CE with logits and softmax'):
        Node.__init__(self, [x, y], name)

    def forward(self, kvargs):
        x, y = [node.value for node in self.inbound_nodes]
        probs = softmax(x)
        self.cache = {'y_hat': probs}
        self.value = np.average(-np.log(probs[:, y]))

    def backward(self, kvargs):
        # last node
        assert not self.outbound_nodes

        self.gradients = {
            n: np.zeros_like(n.value) for n in self.inbound_nodes
        }

        # get values from cache
        x, y = [node for node in self.inbound_nodes]
        y_hat = self.cache['y_hat']
        y_one_hot = np.eye(y_hat.shape[1])[y.value]

        self.gradients[x] = y_hat - y_one_hot
