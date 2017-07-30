"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from MiniFlow.miniflow import *
import numpy as np

def test1():
    x, y, z  = Input(), Input(), Input()
    inputs, weights, bias = Input(), Input(), Input()

    f = Add(x, y, z)
    g = Mul(x, y, z)
    h = LinearV(inputs, weights, bias)

    feed_dict = {x: 10, y: 5, z:4}
    feed_dict2 = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: 2
    }

    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(f, sorted_nodes)
    output2 = forward_pass(g, sorted_nodes)

    graph = topological_sort(feed_dict2)
    output3 = forward_pass(h, graph)

    # NOTE: because topological_sort set the values for the `Input` nodes we could also access
    # the value for x with x.value (same goes for y).
    print("{} + {} + {}  = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
    print("{} x {} x {}  = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output2))
    print(output3) # should be 12.7 with this example

def test3():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    """
    Output should be:
    [[-9., 4.],
    [-9., 4.]]
    """
    print(output)

def test4():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)
    g = Sigmoid(f)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)

    """
    Output should be:
    [[  1.23394576e-04   9.82013790e-01]
     [  1.23394576e-04   9.82013790e-01]]
    """
    print(output)

def test_mse():
    y, a = Input(), Input()
    cost = MSE(y, a)

    y_ = np.array([1, 2, 3])
    a_ = np.array([4.5, 5, 10])

    feed_dict = {y: y_, a: a_}
    graph = topological_sort(feed_dict)
    # forward pass
    forward_pass(cost, graph)

    """
    Expected output
    
    23.4166666667
    """
    print(cost.value)

def test_bp():
    X, W, b = Input(), Input(), Input()
    y = Input()
    f = Linear(X, W, b)
    a = Sigmoid(f)
    cost = MSE(y, a)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2.], [3.]])
    b_ = np.array([-3.])
    y_ = np.array([1, 2])

    feed_dict = {
        X: X_,
        y: y_,
        W: W_,
        b: b_,
    }

    graph = topological_sort(feed_dict)
    forward_and_backward(graph)
    # return the gradients for each Input
    gradients = [t.gradients[t] for t in [X, y, W, b]]

    """
    Expected output

    [array([[ -3.34017280e-05,  -5.01025919e-05],
           [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
           [ 1.9999833]]), array([[  5.01028709e-05],
           [  1.00205742e-04]]), array([ -5.01028709e-05])]
    """
    print(gradients)

test_bp()


