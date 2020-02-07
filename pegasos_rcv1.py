# This file documents the pegasos SVM algorithm as optimized for the rcv1 dataset.

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
