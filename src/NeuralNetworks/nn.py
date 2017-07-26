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

test4()
