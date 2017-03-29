# ===============================================================================================================
# CS5542 Big Data Analytics & Application Lab
# Assignment #7 - TensorFlow
# Linear Regression in own dataset + Training/Testing Cost
# 20 Chia-Hui Amy Lin
# Reference : http://aqibsaeed.github.io/2016-07-07-TensorflowLR/
# ===============================================================================================================

# -------------------------------------------- [ Import Libraries ] ---------------------------------------------
from __future__ import print_function
import tensorflow as tf
import numpy as np
import numpy
import matplotlib.pyplot as plt

# ------------------------------------ [ Train Datasets + Linear Regression Model ] -----------------------------
rng = np.random

# Training Data Sets
trX_list = []
trY_list = []

with open("train_x_data", "r") as file1:
    trX_list.append(file1.read().splitlines())
trX_list = trX_list[0]
trX_list = filter(lambda x:x != '', trX_list)
trX_list = [int(i) for i in trX_list]
trX = numpy.asarray(trX_list)

with open("train_y_data", "r") as file2:
    trY_list.append(file2.read().splitlines())
trY_list = trY_list[0]
trY_list = filter(lambda x:x != '', trY_list)
trY_list = [int(i) for i in trY_list]
trY = numpy.asarray(trY_list)

# Create symbolic variables ( Inputs for tf Grapth )
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create a shared variable for the weight matrix ( Set Model Weights & Bias )
w = tf.Variable(rng.randn(), name="weights")
b = tf.Variable(rng.randn(), name="bias")

# Prediction function ( Build a Linear Model )
y_model = tf.add(tf.multiply(X, w), b)

# ------------------------ [ Cost & Optimizer - Reduce error to improve accuracy ] ------------------------------
# Mean squared error
cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*100)

# Construct an optimizer to minimize cost and fit line to data ( Gradient Descent Optimizer )
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# -------------------------------- [ Initialize the session & global variables ] --------------------------------
# Launch the graph in a session
sess = tf.Session()

# Initialize the variables
init = tf.global_variables_initializer()

# Initialize variables
sess.run(init)

# ------------------------------------------------- [ Training ] -------------------------------------------------
training_epochs = 100
training_cost_track = np.empty(shape=[1], dtype=float)

for i in range(100):  # 100 iterations
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})  # Use the feed dictionary
        training_cost_track = np.append(training_cost_track, sess.run(cost, feed_dict={X: trX, Y: trY}))
training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})

# Output optimization end code and Training Cost with Weights and Bias
print("<<<<< Optimization Finished! >>>>>")
print("------------------------------------------------------------------------")
print("Training cost = ", training_cost, "\n", "Weights( W ) = ", sess.run(w), "\n ", "  Bias( b ) = ", sess.run(b))

# Plot Training Cost
plt.plot(range(len(training_cost_track)), training_cost_track)
plt.axis([0, training_epochs, 0, np.max(training_cost_track)])
plt.title("Training Cost")
plt.show()

# Plot Training Result
plt.plot(trX, trY, 'ro', label='Original data')
plt.plot(trX, sess.run(w) * trX + sess.run(b), label='Fitted line')
plt.legend()
plt.title("Training Result")
plt.show()

print("------------------------------------------------------------------------")

# -------------------------------------------------- [ Testing ] --------------------------------------------------
# Set Testing datasets
testX_list = []
testY_list = []

with open("test_x_data", "r") as file3:
    testX_list.append(file3.read().splitlines())
testX_list = testX_list[0]
testX_list = filter(lambda x:x != '', testX_list)
testX_list = [int(i) for i in testX_list]
test_X = numpy.asarray(testX_list)
file3.close()

with open("test_y_data", "r") as file4:
    testY_list.append(file4.read().splitlines())
testY_list = testY_list[0]
testY_list = filter(lambda x:x != '', testY_list)
testY_list = [int(i) for i in testY_list]
test_Y = numpy.asarray(testY_list)
file4.close()

# Output Testing Mean Square Loss Comparison + Testing Cost + Absolute Mean Square Loss Difference
print("<<<<< Testing... ---> Mean Square Loss Comparison >>>>>")
print("------------------------------------------------------------------------")
testing_cost = sess.run(
    tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
print("Testing cost  = ", testing_cost)  # Use existing model to calculate the cost
print("------------------------------------------------------------------------")
print("[ Absolute Mean Square Loss Difference:", abs(training_cost - testing_cost), " ]")
print("------------------------------------------------------------------------")

# Plot Testing result
plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(trX, sess.run(w) * trX + sess.run(b), label='Fitted line')
plt.legend()
plt.title("Testing Result")
plt.show()