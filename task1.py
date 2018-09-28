import network2
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# read in the data
# logging.info("READING IN DATA...")

# for reading in normal dataset
# training, validation, test = network2.load_data_wrapper("data/mnist.pkl.gz")





### I WILL ADD AND COMMENT OUT SECTIONS OF CODE BASED ON WHICH TASKS I AM TRYING TO EXECUTE FOR ANY GIVEN ITERATION ###



# Task 1 - Experimenting with BPNN

## Task 1.1 - Effect of cost function with default network structure [784, 10]
### - Quadratic cost function with sigmoid activation function; plot convergence
# logging.info("TASK 1.1 A...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_1a.csv", 'w')
#
# for _ in range(3):
#     network = network2.Network([784, 10], cost=network2.QuadraticCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = "Iteration {}\n".format(_)
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}\n\n".format(results)
#     f.write(buff)
#     f.flush()
# f.close()


### - Cross entropy cost function with sigmoid activation function; plot convergence
# logging.info("TASK 1.1 B...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_1b.csv", 'w')
#
# for _ in range(3):
#     network = network2.Network([784, 10], cost=network2.CrossEntropyCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = "Iteration {}\n".format(_)
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}\n\n".format(results)
#     f.write(buff)
#     f.flush()
#
# f.close()


### - Log-likelihood cost function with softmax activation function; plot convergence
# logging.info("TASK 1.1 C...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_1c.csv", 'w')
#
# for _ in range(3):
#     network = network2.Network([784, 10], cost=network2.LogLikelihoodCost, output_activation=network2.SoftmaxActivation)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = "Iteration {}\n".format(_)
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}\n\n".format(results)
#     f.write(buff)
#     f.flush()
#
# f.close()



## Task 1.2 - Effect of regularization with default network structure [784, 10], no hidden layers, and cross entropy
### - Add L2 normalization on the cost function; plot convergence
# logging.info("TASK 1.2 A...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_2a.csv", 'w')
#
# for l2 in [0.01, 0.1, 1, 10]:
#     network = network2.Network([784, 10], cost=network2.CrossEntropyCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, lmbda=l2, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = str(l2) + '\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}".format(results) + '\n\n'
#     f.write(buff)
#     f.flush()
#
# f.close()


### - Add L1 normalization on the cost function; plot convergence
# logging.info("TASK 1.2 B...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_2b.csv", 'w')
#
# for l1 in [0.01, 0.1, 1, 10]:
#     network = network2.Network([784, 10], cost=network2.CrossEntropyCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, gmma=l1, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = str(l1) + '\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}".format(results) + '\n\n'
#     f.write(buff)
#     f.flush()
#
# f.close()


### - L1 normalization; expanded training set with affine transforms; plot convergence
# read in the data
logging.info("READING IN DATA...")

# for reading in normal dataset
training, validation, test = network2.load_data_wrapper("data/mnist_expanded.pkl.gz")

# logging.info("TASK 1.2 C...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_2c.csv", 'w')
#
# for _ in range(3):
#     network = network2.Network([784, 10], cost=network2.CrossEntropyCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, gmma=1, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     logging.info("WRITING RESULTS...")
#     buff = "Iteration {}\n".format(_)
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}".format(results) + '\n\n'
#     f.write(buff)
#     f.flush()
#
# f.close()



## Task 1.3 - Effect of hidden layers; cross entropy; L1 normalization; expanded training set
### - Add one hidden layer with 30 nodes [784, 30, 10]; plot convergence
logging.info("TASK 1.3 A...")
logging.info("INITIALIZING NETWORK...")

f = open("task1_3a.csv", 'w')

for _ in range(3):
    network = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    logging.info("TRAINING NETWORK...")
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, gmma=1, evaluation_data=validation)

    logging.info("EVALUATING RESULTS...")
    results = network.accuracy(test)

    logging.info("WRITING RESULTS...")
    buff = "Iteration {}\n".format(_)
    f.write(buff)
    buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
    f.write(buff)
    buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
    f.write(buff)
    buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
    f.write(buff)
    buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
    f.write(buff)
    buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
    f.write(buff)
    buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
    f.write(buff)
    buff = "test_acc,{}".format(results) + '\n\n'
    f.write(buff)
    f.flush()

f.close()


### - Add two hidden layers with 30 nodes [784, 30, 30, 10]; plot convergence; plot change rate of each weight in hidden layers
# logging.info("TASK 1.3 B...")
# logging.info("INITIALIZING NETWORK...")
#
# f = open("task1_3b.csv", 'w')
#
# for _ in range(3):
#     network = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost)
#
#     logging.info("TRAINING NETWORK...")
#     evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, weight_change = network.SGD(training, 100, 100, 0.9, gmma=1, evaluation_data=validation)
#
#     logging.info("EVALUATING RESULTS...")
#     results = network.accuracy(test)
#
#     weight_np = np.array(weight_change)
#
#     logging.info("WRITING RESULTS...")
#     buff = "Iteration {}\n".format(_)
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
#     f.write(buff)
#     buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
#     f.write(buff)
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
#     f.write(buff)
#     buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
#     f.write(buff)
#     buff = "test_acc,{}".format(results) + '\n\n'
#     f.write(buff)
#     f.flush()
#
#     buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
#     f.write(buff)
#     for i in range(weight_np.shape[1]):
#         buff = "w_change_{},".format(i) + ','.join([str(j) for j in weight_change[:][i]]) + '\n'
#         f.write(buff)
#
#     f.flush()
#
# f.close()


### - (692) L1 normalization; expanded training set; dropout (several %-ages); plot convergence
