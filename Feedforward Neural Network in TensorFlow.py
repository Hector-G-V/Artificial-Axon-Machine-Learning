"""
Feed-Forward Neural Network with one hidden layer is trained with TensorFlow tools.
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


def neural_network_model(x, n_nodes_hl, n_classes):
    """
    The neural network model with one hidden layer.
    :param x: Input layer data.
    :param n_nodes_hl: Number of nodes in the hidden layer.
    :param n_classes: Number of classes.
    :return: Output layer, hidden layer weights & biases, output layer weights & biases.
    """

    # Hidden layer matrices
    hl_matrices = {"weights": tf.Variable(tf.random_uniform([784, n_nodes_hl], minval=-0.05, maxval=0.05)),
                   "biases": tf.Variable(tf.random_uniform([n_nodes_hl], minval=-0.05, maxval=0.05))}
    # Output layer matrices
    ol_matrices = {"weights": tf.Variable(tf.random_uniform([n_nodes_hl, n_classes], minval=-0.05, maxval=0.05)),
                   "biases": tf.Variable(tf.random_uniform([n_classes], minval=-0.05, maxval=0.05))}

    hidden_layer = tf.matmul(x, hl_matrices['weights']) + hl_matrices['biases']
    hidden_layer = tf.keras.activations.sigmoid(10 ** 4 * (hidden_layer - 0.8))  # Activation.

    output_layer = tf.matmul(hidden_layer, ol_matrices['weights']) + ol_matrices['biases']

    return output_layer, hl_matrices, ol_matrices


def train_neural_network(x, y, n_nodes_hl, n_classes, n_epochs, batch_size):
    """
    Trains the neural network.
    :param x: Input layer data.
    :param y: Test labels.
    :param n_nodes_hl: Number of nodes in the hidden layer.
    :param n_classes: Number of classes.
    :param n_epochs: Number of epochs.
    :param batch_size: Batch size.
    :return: Trained hidden and output layer matrices.
    """

    # Tensorflow optimizer and cost functions.
    prediction = neural_network_model(x, n_nodes_hl, n_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction[0], labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = np.heaviside(batch_x - 0.5, 1)  # Pre-processing.
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

            print("Epoch", epoch, 'completed out of', n_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction[0], 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: np.heaviside(mnist.test.images - 0.5, 1), y: mnist.test.labels}))

        # The trained matrices.
        trained_hidden_layer, trained_output_layer = sess.run(prediction[1]), sess.run(prediction[2])

        return trained_hidden_layer, trained_output_layer


if __name__ == '__main__':

    # Args
    n_nodes = 100  # Number of hidden layer nodes.
    n_class = 10  # Number of classes.
    epochs = 100  # Number of epochs.
    n_batch = 100  # Batch size.

    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float")

    # Collect the matrices
    matrices = train_neural_network(X, Y, n_nodes, n_class, epochs, n_batch)
