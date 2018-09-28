import network2
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# read in the data
logging.info("READING IN DATA...")

# for reading in normal dataset
training, validation, test = network2.load_data_wrapper("data/mnist_expanded.pkl.gz")

logging.info("TASK 1.3 C...")
logging.info("INITIALIZING NETWORK...")

f = open("task1_3c.csv", 'w')

for do in [.2, .4, .6, .8]:
    network = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost, dropout=do)

    logging.info("TRAINING NETWORK...")
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, _ = network.SGD(training, 100,
                                                                                                        100, 0.9,
                                                                                                        gmma=1,
                                                                                                        evaluation_data=validation)

    logging.info("EVALUATING RESULTS...")
    results = network.accuracy(test)

    logging.info("WRITING RESULTS...")
    buff = "{}\n".format(do)
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
