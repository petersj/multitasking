import tensorflow as tf
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian
import time
import random

def gen_net(n_input_dims, n_hidden, n_output_dims, n_tasks):

    n_input_units = n_input_dims*2
    n_output_units = n_output_dims*2

    inputs = tf.placeholder(tf.float32, [None, n_input_units])

    task = tf.placeholder(tf.float32, [None, n_tasks])

    W_input_hidden = tf.Variable(tf.random_uniform([n_input_units, n_hidden], minval=-.1, maxval=.1))
    W_task_hidden = tf.Variable(tf.random_uniform([n_tasks, n_hidden], minval=-.1, maxval=.1))
    b_hidden = tf.constant([-2.]*n_hidden)
    hidden = tf.sigmoid(tf.matmul(inputs, W_input_hidden)+tf.matmul(task, W_task_hidden)+b_hidden)

    W_hidden_output = tf.Variable(tf.random_uniform([n_hidden, n_output_units], minval=-.1, maxval=.1))
    W_task_output = tf.Variable(tf.random_uniform([n_tasks, n_output_units], minval=-.1, maxval=.1))
    b_output = tf.constant([-2.]*n_output_units)
    outputs = tf.sigmoid(tf.matmul(hidden, W_hidden_output)+tf.matmul(task, W_task_output)+b_output)

    outputs_ = tf.placeholder(tf.float32, [None, n_output_units]) # desired output
    error = 0.5 * tf.reduce_mean(tf.square((outputs_ - outputs))) # loss function
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(error) # update

    return inputs, task, W_input_hidden, W_task_hidden, b_hidden, hidden, W_hidden_output, W_task_output, b_output, outputs, outputs_, error, train_step

def get_task_dims(n_input_dims, n_output_dims, task):
    return [task//n_output_dims, task%n_output_dims]

def gen_training_batch(n_input_dims, n_output_dims, n_possible_tasks, task_ids):
    n_input_units = n_input_dims*2
    n_output_units = n_output_dims*2
    n_inputs = 2**n_input_dims
    idx_list = []
    for i in range(n_input_dims):
        idx_list.append([i*2, i*2+1])
    idx_list = cartesian(idx_list)

    inputs_list = np.zeros((idx_list.shape[0], n_input_units))
    for i in range(idx_list.shape[0]):
        inputs_list[i, :][idx_list[i]] = 1

    inputs_list = np.tile(inputs_list, (len(task_ids), 1))

    task_list = np.zeros((n_inputs*len(task_ids), n_possible_tasks))
    for i in range(len(task_ids)):
        task_list[i*n_inputs:(i*n_inputs+n_inputs), task_ids[i]] = 1

    outputs_list = np.zeros((n_inputs*len(task_ids), n_output_units))
    for i in range(len(task_ids)):
        for j in range(len(task_ids[i])):
            input_dim, output_dim = get_task_dims(n_input_dims, n_output_dims, task_ids[i][j])
            input_pattern = inputs_list[i*n_inputs:(i*n_inputs+n_inputs), input_dim*2:input_dim*2+2]
            outputs_list[i*n_inputs:(i*n_inputs+n_inputs), output_dim*2:output_dim*2+2] = input_pattern

    return inputs_list, task_list, outputs_list

def gen_task_list(n_input_dims, n_output_dims, pathway_overlap):
    if pathway_overlap > n_output_dims:
        raise ValueError("Pathway overlap can not be greater than the number of output dimensions")
    task_list = []
    for i in range(n_input_dims):
        task_list.append([i*n_output_dims + i])
        for j in range(pathway_overlap-1):
            next_task = []
            for k in range(n_output_dims):
                if [i*n_output_dims + k] not in task_list:
                    next_task.append([i*n_output_dims + k])
            task_list.append(random.choice(next_task))
    return task_list


def main():
    n_input_dims = 6
    n_output_dims = 6
    n_hidden = 200
    pathway_overlap = 2
    n_possible_tasks = n_input_dims * n_output_dims
    task_list = gen_task_list(n_input_dims, n_output_dims, pathway_overlap)
    iterations = 500
    thresh = 0.0001

    inputs, task, W_input_hidden, W_task_hidden, b_hidden, hidden, W_hidden_output, W_task_output, b_output, outputs, outputs_, error, train_step = gen_net(n_input_dims, n_hidden, n_output_dims, n_possible_tasks)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    training_batch_inputs, training_batch_task, training_batch_outputs = gen_training_batch(n_input_dims, n_output_dims, n_possible_tasks, task_list)

    n_trials = training_batch_inputs.shape[0]
    shuffle = np.random.permutation(n_trials)
    rand_training_batch_inputs = training_batch_inputs[shuffle, :]
    rand_training_batch_task = training_batch_task[shuffle, :]
    rand_training_batch_outputs = training_batch_outputs[shuffle, :]

    running_MSE_mean = np.zeros((iterations,))
    for i in range(iterations):
        MSE = np.zeros((n_trials,))
        for j in range(n_trials):
            train_step.run(feed_dict = {inputs:[training_batch_inputs[j, :]], task:[training_batch_task[j, :]], outputs_:[training_batch_outputs[j, :]]})
            MSE[j] = error.eval(feed_dict = {inputs:[training_batch_inputs[j, :]], task:[training_batch_task[j, :]], outputs_:[training_batch_outputs[j, :]]})
        running_MSE_mean[i] = np.mean(MSE)
        print(i)
    plt.plot(running_MSE_mean)
    plt.show()



if __name__ == '__main__':
    main()