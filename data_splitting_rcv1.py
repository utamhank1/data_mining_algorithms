# This script takes the rcv1 dataset (www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf) of 800,000 categorized news
# stories and splits it into training and test data.

# Import libraries
import numpy as np
from sklearn.datasets import fetch_rcv1


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

    # Save the training and test datasets to disk.
    np.save('test_data_rcv1.npy', test_data)
    np.save('test_label_rcv1', test_label)
    np.save('training_data_rcv1', training_data)
    np.save('training_label_rcv1', training_label)


if __name__ == '__main__':
    main()
