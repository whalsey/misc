import network2
import logging
logging.basicConfig(level=logging.DEBUG)

# read in the data
logging.info("READING IN DATA...")

# for reading in normal dataset
training, validation, test = network2.load_data_wrapper("data/mnist.pkl.gz")

# for reading in expanded dataset
# training, validation, test = load_data_wrapper("data/mnist_expanded.pkl.gz")




### I WILL ADD AND COMMENT OUT SECTIONS OF CODE BASED ON WHICH TASKS I AM TRYING TO EXECUTE FOR ANY GIVEN ITERATION ###



# Task 1 - Experimenting with BPNN

## Task 1.1 - Effect of cost function with default network structure [784, 10]
### - Quadratic cost function with sigmoid activation function; plot convergence
# logging.info("TASK 1.1 A...")
# logging.info("INITIALIZING NETWORK...")
# network = network2.Network([784, 10], cost=network2.QuadraticCost)
#
# logging.info("TRAINING NETWORK...")
# evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, evaluation_data=validation)
#
# logging.info("EVALUATING RESULTS...")
# results = network.accuracy(test)
#
# logging.info("WRITING RESULTS...")
# f = open("task1_1a.csv", 'w')
# buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
# f.write(buff)
# buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
# f.write(buff)
# buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
# f.write(buff)
# buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
# f.write(buff)
# buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
# f.write(buff)
# buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
# f.write(buff)
# buff = "test_acc,{}".format(results)
# f.write(buff)
# f.close()


### - Cross entropy cost function with sigmoid activation function; plot convergence
# logging.info("TASK 1.1 B...")
# logging.info("INITIALIZING NETWORK...")
# network = network2.Network([784, 10], cost=network2.QuadraticCost)
#
# logging.info("TRAINING NETWORK...")
# evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, evaluation_data=validation)
#
# logging.info("EVALUATING RESULTS...")
# results = network.accuracy(test)
#
# logging.info("WRITING RESULTS...")
# f = open("task1_1b.csv", 'w')
# buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
# f.write(buff)
# buff = "eval_cost," + ','.join([str(i) for i in evaluation_cost]) + '\n'
# f.write(buff)
# buff = "train_cost," + ','.join([str(i) for i in training_cost]) + '\n\n'
# f.write(buff)
# buff = "epoch," + ','.join([str(i) for i in range(100)]) + '\n'
# f.write(buff)
# buff = "eval_acc," + ','.join([str(i) for i in evaluation_accuracy]) + '\n'
# f.write(buff)
# buff = "train_acc," + ','.join([str(i) for i in training_accuracy]) + '\n\n'
# f.write(buff)
# buff = "test_acc,{}".format(results)
# f.write(buff)
# f.close()


### - Log-likelihood cost function with softmax activation function; plot convergence



## Task 1.2 - Effect of regularization with default network structure [784, 10], no hidden layers, and cross entropy
### - Add L2 normalization on the cost function; plot convergence
logging.info("TASK 1.2 A...")
logging.info("INITIALIZING NETWORK...")

f = open("task1_2a.csv", 'w')

for l2 in [0.01, 0.1, 1, 10]:
    network = network2.Network([784, 10], cost=network2.QuadraticCost)

    logging.info("TRAINING NETWORK...")
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = network.SGD(training, 100, 100, 0.9, lmbda=l2, evaluation_data=validation)

    logging.info("EVALUATING RESULTS...")
    results = network.accuracy(test)

    logging.info("WRITING RESULTS...")
    buff = str(l2) + '\n'
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

f.close()

### - Add L1 normalization on the cost function; plot convergence


### - L1 normalization; expanded training set with affine transforms; plot convergence


### - (692) L1 normalization; expanded training set; dropout (several %-ages); plot convergence


## Task 1.3 - Effect of hidden layers; cross entropy; L1 normalization; expanded training set
### - Add one hidden layer with 30 nodes [784, 30, 10]; plot convergence
### - Add two hidden layers with 30 nodes [784, 30, 30, 10]; plot convergence; plot change rate of each weight in hidden layers