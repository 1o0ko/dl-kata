'''
Tests the node implementations
'''
import unittest

from tinyflow.core import value_and_grad
from tinyflow.ops import (
    Input,
    Add,
    Mul,
    MockGrad
)


class OpsTest(unittest.TestCase):
    ''' Tests operations from the ops module '''

    def test_mul(self):
        ''' Tests Mul operation '''
        x, y, z = Input(), Input(), Input()

        g = Mul(x, y)
        h = Mul(x, z)

        f = Add(g, h)

        # MockGrad is just here so we can pass a fake gradient backwards.
        mock = MockGrad(f)

        # values for Input and mockGrad nodes.
        feed_dict = {x: 3, y: 4, z: -5, mock: 0.5}

        loss, grads = value_and_grad(mock, feed_dict, (x, y, z))

        # print(loss, grads)
        self.assertEqual(loss, -3)
        self.assertEqual(grads, [-0.5, 1.5, 1.5])


if __name__ == '__main__':
    unittest.main()
