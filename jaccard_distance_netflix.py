# This script creates a histogram of the jaccard distances between individual users' movie ratings in the netflix
# dataset. This scrip also outputs a sparse table of the jaccard distances.


# Jaccard distance calculation function
def jaccard(col1, col2):
    return (col1 ^ col2).sum() / (col1 | col2).sum()


def main():
    # Import necessary libraries
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    from scipy import sparse

    # Load in the table of users and ratings that are outputted from data_cleaning_netflix.py
    table = np.load("table.npy")

    # Initialize empty matrix.
    jaccardDistance = []

    # Iterate over the table of user ratings and calculate the jaccard distances from each neighbor.
    for i in range(0, 10000):
        columnNum1 = random.randint(0, table.shape[1] - 1)
        columnNum2 = random.randint(0, table.shape[1] - 1)
        result = jaccard(table[:, columnNum1], table[:, columnNum2])
        jaccardDistance.append(result)

    # Generate a plot of the count of the jaccard distances.
    fig = plt.figure()
    plt.hist(jaccardDistance, range=(0, 1), bins=20)
    plt.title('Histogram of Jaccard Distance')
    plt.xlabel('Jaccard Distance')
    plt.ylabel('Count')
    plt.show()
    fig.savefig('Histogram of Jaccard Distance', dpi=fig.dpi)

    # Save data to a sparse table (to save memory).
    sparse_table = sparse.coo_matrix(table)
    np.save("sparse_table.npy", sparse_table)


if __name__ == "__main__":
    main()
