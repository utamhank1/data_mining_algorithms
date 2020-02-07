# This script implements the adagrad and pegasos algorithms on the rcv1 training and test data and outputs plots of
# their performance.

# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import fetch_rcv1


def Pegasos_Alg(training_data, training_label, lamda, T, k):
    m, n = training_data.shape
    w = np.zeros((T + 1, n))
    A_t_star = []

    for t in range(1, T):
        # choose a subset indexes from training data:
        A_t = random.sample(range(0, m), k)

        # get At+ subset by y<x,wt> < 1
        sum_A_t_star = 0
        for i in A_t:
            if training_data[i] * w[t] * training_label[i] < 1:
                sum_A_t_star += training_label[i] * training_data[i]
        ita_t = 1 / (t * lamda)
        gradient = lamda * w[t] - (sum_A_t_star) / k
        w[t + 1] = w[t] - ita_t * gradient
        w_eDis = np.dot(w[t + 1], w[t + 1].T)
        if w_eDis > 1 / lamda:
            w[t + 1] = w[t + 1] / ((np.dot(w[t + 1], w[t + 1].T) * lamda) ** 0.5)

    return w


def getPredictAccuracy(w, data, label):
    predict_result = data * w * label
    right_predict_count = len(np.where(predict_result > 0)[0])
    accuracy = right_predict_count / len(label)
    return accuracy


def plotAccuracy(SVM, lamda, k):
    training_accuracy = []

    for w in SVM:
        training_accuracy.append(getPredictAccuracy(w, training_data, training_label))
    epoch_count = range(1, len(training_accuracy) + 1)
    plt.plot(epoch_count, training_accuracy, 'b')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    print('When lamda is:', lamda, '; Batch size is:', k, '; The best training accuracy is:', max(training_accuracy),
          'at iteration:', training_accuracy.index(max(training_accuracy)))
    plt.show()


def plotBothAccuracy(SVM, lamda, k):
    training_accuracy = []
    test_accuracy = []
    for w in SVM:
        training_accuracy.append(getPredictAccuracy(w, training_data, training_label))
        test_accuracy.append(getPredictAccuracy(w, test_data, test_label))
    epoch_count = range(1, len(test_accuracy) + 1)
    plt.plot(epoch_count, training_accuracy, 'b')
    plt.plot(epoch_count, test_accuracy, 'r')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    print('When lamda is:', lamda, '; Batch size is:', k, '; The best training accuracy is:', max(training_accuracy),
          'at iteration:', training_accuracy.index(max(training_accuracy)))
    print('The best test accuracy is:', max(test_accuracy), 'at iteration:', test_accuracy.index(max(test_accuracy)))
    plt.show()


def AdaGrad_Alg(training_data, training_label, lamda, T, k):
    m, n = training_data.shape
    w = np.zeros([(T + 1), n])
    s = np.ones([(T + 1), n])

    for t in range(1, T):
        # choose a subset indexes from training data:
        A_t = random.sample(range(0, m), k)
        sum_A_t_star = 0
        for i in A_t:
            if training_data[i] * w[t] * training_label[i] < 1:
                sum_A_t_star += training_label[i] * training_data[i]

        ita_t = 2
        gradient = lamda * w[t] - sum_A_t_star / k

        w[t + 1] = w[t] - np.divide((ita_t * gradient), np.sqrt(s[t]))
        w_eDis = np.dot(w[t + 1], w[t + 1].T)
        if w_eDis > 1 / lamda:
            w[t + 1] = w[t + 1] / ((np.dot(np.multiply(np.sqrt(s[t]), w[t + 1]),
                                           np.multiply(np.sqrt(s[t].T), w[t + 1].T)) * lamda) ** 0.5)
        s[t + 1] = s[t] + np.square(gradient)

    return w


def main():
    # Fetch the rcv1 dataset from sklearn.
    rcv1 = fetch_rcv1()

    # Clean and reformat the dataset.
    target = rcv1['target'].todense()
    label = np.array(target[:, 33]).reshape(1, -1)[0]
    label.dtype = 'int8'
    label[label == 0] = -1

    # Create numpy array of training data.
    training_data = rcv1['data'][0:100000, :]

    # Assign labels to training data.
    training_label = label[0:100000]

    test_data = rcv1['data'][100000:, :]
    test_label = label[100000:]

    # Plot various graphs of adagrad performance for a varety of batch sizes and lambda values on the training data.
    T = 2000
    batchSize_set = [1, 10, 100, 300]
    lamda_set = [0.1, 0.01, 0.001, 0.0001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = Pegasos_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], training_data, training_label)
            # plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    T = 2000
    batchSize_set = [1, 10, 100, 300]
    lamda_set = [0.01, 0.001, 0.0001, 0.00001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = Pegasos_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], training_data, training_label)
            plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    T = 10000
    batchSize_set = [1, 10, 100, 300]
    lamda_set = [0.01, 0.001, 0.0001, 0.00001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = AdaGrad_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], training_data, training_label)
            # plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    T = 2000
    batchSize_set = [1, 10, 100]
    lamda_set = [0.001, 0.0001, 0.00001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = AdaGrad_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], training_data, training_label)
            plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    # Plot various graphs of adagrad performance for a variety of batch sizes and lambda values on the test data.
    T = 10000
    batchSize_set = [300]
    lamda_set = [0.0001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = AdaGrad_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], test_data, test_label)
            plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    # Run pegasos algorithm and plot performance based on optimal batchsize, T, and lambda values obtained from adagrad.
    T = 10000
    batchSize_set = [300]
    lamda_set = [0.0001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = Pegasos_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], test_data, test_label)
            plotAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    # Show training and test accuracy of Pegasos.
    T = 2000
    batchSize_set = [300]
    lamda_set = [0.0001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = Pegasos_Alg(training_data, training_label, lamda, T, k)
            accuracy = getPredictAccuracy(SVM[T], test_data, test_label)
            plotBothAccuracy(SVM, lamda, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %0.3f' % accuracy)
            print("done in %0.3fs" % (time() - t0))

    # Show training and test accuracy of AdaGrad.
    T = 2000
    batchSize_set = [300]
    lamda_set = [0.0001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = AdaGrad_Alg(training_data, training_label, lamda, T, k)
            plotBothAccuracy(SVM, lamda, k)
            print("done in %0.3fs" % (time() - t0))

    # Show best training and test accuracy of Pegasos
    T = 10000
    batchSize_set = [300]
    lamda_set = [0.00001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = Pegasos_Alg(training_data, training_label, lamda, T, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %.2f%%' % (
                          getPredictAccuracy(SVM[T], training_data, training_label) * 100))
            print('The test accuracy is: %.2f%%' % (getPredictAccuracy(SVM[T], test_data, test_label) * 100))
            print("done in %0.3fs" % (time() - t0))

    # Show best training and test accuracy of AdaGrad.
    T = 10000
    batchSize_set = [300]
    lamda_set = [0.00001]
    for k in batchSize_set:
        for lamda in lamda_set:
            t0 = time()
            SVM = AdaGrad_Alg(training_data, training_label, lamda, T, k)
            print('When lamda is:', lamda, '; Batch size is:', k, '; Number of iteration is:', T,
                  '; The training accuracy is: %.2f%%' % (
                          getPredictAccuracy(SVM[T], training_data, training_label) * 100))
            print('The test accuracy is: %.2f%%' % (getPredictAccuracy(SVM[T], test_data, test_label) * 100))
            print("done in %0.3fs" % (time() - t0))


if __name__ == "__main__":
    main()
