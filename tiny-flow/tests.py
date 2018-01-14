'''
Tests the node implementations
'''
import unittest
import numpy as np

from tinyflow.math import softmax
from tinyflow.core import value_and_grad
from tinyflow.ops import (
    Input,
    Add,
    Mul,
    Linear,
    Sigmoid,
    CrossEntropyWithLogits,
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

    def test_cross_enrtopy_with_softmax(self):
        x_in, y_in = Input(), Input()

        f = CrossEntropyWithLogits(x_in, y_in)

        # values to feed input nodes
        x = np.array([[0.5, 1., 1.5]])

        # in this example we have a choice 3 classes (x has 3 columns)
        # so our label can one of 0,1,2. It's 1 in this case.
        y = np.array([[1]])

        feed_dict = {x_in: x, y_in: y}
        loss, grads = value_and_grad(f, feed_dict, wrt=[x_in])

        # print(loss, grads)
        # Look at the expected value of softmax(x) and the expected value of the gradient with
        # respect to x
        self.assertTrue(np.allclose(
           softmax(x), [[0.1863, 0.3072,  0.5064]], atol=1.e-4))

        self.assertTrue(np.allclose(
            loss, 1.1802, atol=1.e-4))

        self.assertTrue(np.allclose(
            grads, np.array([[0.1863, -0.6928,  0.5064]]), atol=1.e-4))

        # TODO: add test case with several rows


if __name__ == '__main__':
    unittest.main()
