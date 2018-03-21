###################################################
#
#    NOTE: This code is broken on purpose
#
###################################################

import csv
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load data
reader = csv.reader(open("linear.csv", "r"), delimiter=",")
x = list(reader)
data = np.array(x).astype('int')

# show images
fig, axes = plt.subplots(6, 20, figsize=(18, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
   ax.imshow(-1 * (data[i,1:].reshape((3,3)) - 255), cmap='gray') 


# split data
Xdata = np.array(data[:,1:])
ydata = np.array(data[:,0]).reshape((data.shape[0], 1)) * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.20)

# Create tensorflow graph

# simple digits shape 3x3 = 9
x = # a place holder

# 0/1
y = # a place holder

# model weights
W = # a variable
b = # a variable

# predictive function
prediction = # W.T x + b

# accuracy
accuracy = # tricky sign bits

#cost
cost = # cost function

# optimizer
optimizer = # gradient descent doesn't work!! Try Adam

# RUN IT!!!
training_epochs = 1000
batch_size = 10

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    last_cost = 1e10
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X_train[i * batch_size : i * batch_size + batch_size, :]
            batch_ys = y_train[i * batch_size : i * batch_size + batch_size, :]
            # Run optimization, cost, and summary
            o, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
        if last_cost <= avg_cost or math.isnan(avg_cost):
            break
        
        last_cost = avg_cost

    print("Optimization Finished!")

    # Calculate accuracy - it is bad on the first go around
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
    
    WFinal = W.eval()
    BFinal = b.eval()

# PREDICT
h = np.array([0, 0, 0, 255, 255, 255, 0, 0, 0])
v = np.array([0, 255, 0, 0, 255, 0, 0, 255, 0])

# should be negative for horizontal lines
np.dot(h, WFinal) + BFinal

# should be positive for vertical lines
np.dot(v, WFinal) + BFinal
