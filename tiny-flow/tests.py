'''
Tests the node implementations
'''
import numpy as np
import unittest

from tinyflow.core import value_and_grad
from tinyflow.ops import (
    Input,
    Add,
    Mul,
    Linear,
    Sigmoid,
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

    def test_linear(self):
        ''' Tests Linear operation '''
        x_in, w_in, b_in = Input(), Input(), Input()
        f = Linear(x_in, w_in, b_in)
        mock = MockGrad(f)

        x = np.array([[-1., -2.], [-1, -2]])
        w = np.array([[2., -3], [2., -3]])
        b = np.array([-3., -5])
        mock_value = np.array([[1., 2.], [3, 4]])

        feed_dict = {x_in: x, w_in: w, b_in: b, mock: mock_value}
        loss, grads = value_and_grad(mock, feed_dict, (x_in, w_in, b_in))

        # print(loss, grads)
        self.assertTrue(np.allclose(loss, np.array([[-9., 4.], [-9., 4.]])))
        self.assertTrue(np.allclose(
            grads[0], np.array([[-4.,  -4.], [-6.,  -6.]])))
        self.assertTrue(np.allclose(
            grads[1], np.array([[-4.,  -6.], [-8., -12.]])))
        self.assertTrue(np.allclose(
            grads[2], np.array([[4., 6.]])))

    def test_sigmoid(self):
        x_in = Input()

        f = Sigmoid(x_in)
        mock = MockGrad(f)

        x = np.array([-10., 0, 10])

        feed_dict = {x_in: x, mock: 0.5}
        loss, grads = value_and_grad(mock, feed_dict, [x_in])

        # print(loss, grads)
        self.assertTrue(np.allclose(
            loss, np.array([0., 0.5, 1.]), atol=1.e-4))

        self.assertTrue(np.allclose(
            grads, np.array([0., 0.125, 0.]), atol=1.e-4))

if __name__ == '__main__':
    unittest.main()
