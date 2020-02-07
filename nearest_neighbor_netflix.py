# This script finds the nearest neighbor of a queried user. First, we take the table outputted from the script
# jaccard_distance_netflix and parse it to another data structure more conducive to deriving the nearest neighbor
# of each user. Two users are defined as "close" if their Jaccard distances are below 3.5.In order to find similarity
# between two users, it is necessary to construct a signature matrix derived from a characteristic matrix through a
# technique known as MinHashing.

# Import libraries
import numpy as np
import random
from scipy import sparse
import matplotlib.pyplot as plt
import pickle
from itertools import combinations



# Helper function to compute the jaccard distance of two users.
def jaccard(col1, col2):
    return (col1 ^ col2).sum() / (col1 | col2).sum()


# Helper function to determine if two users are considered "similar".
def jaccard_filter(pair, table):
    a, b = pair
    j_dist = jaccard(table[:, a], table[:, b])
    return j_dist < 0.35


# Minhashing algorithm to help detect user similarity.
def createbuckets(b, r, signature_matrix, arr_cols):
    array_buckets = []
    for i in range(b):
        array_buckets.append(dict())

    for i in range(b):
        buckets = array_buckets[i]
        band = signature_matrix[(i * r):((i + 1) * r), :]
        for k in range(arr_cols):
            key = hash(band[:, k].tostring())
            if key in buckets:
                buckets[key].append(k)
            else:
                buckets.update({key: [k]})
    buckets = array_buckets
    return buckets


def main():
    # Load in the table from the previous script.
    table = np.load("table.npy")
    arr_rows, arr_cols = table.shape

    # save table to a python dictionary.
    dictionary_pos = dict()
    for col in range(arr_cols):
        dictionary_pos.update({col: np.where(table[:, col] == 1)})
    with open('dictionary_pos.pkl', 'wb') as handle:
        pickle.dump(dictionary_pos, handle)

    # m, R being the size of the table, we select random coefficients a and b to hash the characteristic matrix of the
    # netflix user dataset.  Instead of operating on the entire characteristic matrix, we only need to operate on the
    # information that we are interested in. In this case, we create a python dictionary where the
    # dictionary key rows correspond to column values in the characteristic matrix that have value 1
    # and values of the keys correspond to columns that have value 1. Similarly, we create another
    # python dictionary with the inverse relationship (keys correspond to columns in characteristic that
    # have value 1 and values of keys correspond to row numbers that have value 1). This
    # representation is more helpful and will help us derive the similarity of users.
    m = 500
    R = 4507
    coeff_a = []
    for i in range(m):
        coeff_a.append(random.randint(0, R))

    coeff_b = []
    for i in range(m):
        coeff_b.append(random.randint(0, R))
    coeff_a = np.array(coeff_a).reshape(m, 1)
    coeff_b = np.array(coeff_b).reshape(m, 1)

    signature_matrix = np.zeros((m, arr_cols))
    for key in range(len(dictionary_pos)):
        signature_matrix[:, key] = np.amin((coeff_a * np.array(pst_dict[key]) + coeff_b) % R, axis=1)

    # Number of buckets and number of rows for minhashing algorithm.
    b = 50
    r = 10
    buckets = createbuckets(b, r, signature_matrix, arr_cols)

    # for every row value in the dictionary keys, we compute 500 hash functions for that vector.This 500 x 1 vector is
    # then compared to the vectors of the infinite signature matrix with column values in the list of the dictionary
    # for the current row number.
    pair_set = set()
    for i in range(len(buckets)):
        for key, val in buckets[i].items():
            if len(val) > 1:
                comb = combinations(val, 2)
            for j in list(comb):
                pair_set.add(j)
    len(pair_set)


    # Generate Plots to figure out optimal value of m
    m = 500
    r = [2, 5, 10, 20, 50, 100]
    b = [m / i for i in r]
    s = np.arange(0, 1, 0.001)
    for i in range(len(r)):
        plt.plot(s, 1 - (1 - s ** r[i]) ** b[i])
    plt.plot([0.65, 0.65], [0, 1], 'black')

    m = 1000
    r = [2, 5, 10, 20, 50, 100]
    b = [m / i for i in r]
    s = np.arange(0, 1, 0.001)
    for i in range(len(r)):
        plt.plot(s, 1 - (1 - s ** r[i]) ** b[i])
    plt.plot([0.65, 0.65], [0, 1], 'black')

    m = 100
    r = [2, 5, 10, 20, 50, 100]
    b = [m / i for i in r]
    s = np.arange(0, 1, 0.001)
    for i in range(len(r)):
        plt.plot(s, 1 - (1 - s ** r[i]) ** b[i])
    plt.plot([0.65, 0.65], [0, 1], 'black')

    def findNearUser(user):
        r = 10
        b = 50
        # This will give us the signature matrix of the queried user
        signature_matrix_user = np.amin((coeff_a * np.array(user) + coeff_b) % R, axis=1)
        # hash the user into previous bucket
        user_buckets = createbuckets(signature_matrix_user, b, r)
        # find candidate pairs in the bucket
        pair = set()
        # find nearest one
        x, y = pair
        for i in range(len(buckets)):
            for j in range(len(user_buckets)):
                for key, val in buckets[i].items():
                    for key_user, val_user in user_buckets[j].items():
                        if key == key_user:
                            pair.add((val, val_user))
        nearest_neighbors = set(filter(jaccard_filter, pair))
        print(len(pair))
        index = np.amin(j_dist=jaccard(user, table[:, y]))
        return table[:, y]

        user = [0, 0, 1]
        findNearUser(user)


if __name__ == '__main__':
    main()
