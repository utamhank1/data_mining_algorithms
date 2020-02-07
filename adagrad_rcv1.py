# This file documents the adagrad SGD algorithm as optimized for the rcv1 dataset.

def AdaGrad_Alg(training_data, training_label, lamda, T, k):
    m, n = training_data.shape
    w = np.zeros([(T + 1), n])
    s = np.ones([(T + 1), n])
    A_t_star = []

    for t in range(1, T):
        # choose a subset indexes from training data:
        A_t = random.sample(range(0, m), k)
        sum_A_t_star = 0
        for i in A_t:
            if training_data[i] * w[t] * training_label[i] < 1:
                sum_A_t_star += training_label[i] * training_data[i]
        # ita_t = 1/(T**0.5)

        ita_t = 2
        gradient = lamda * w[t] - (sum_A_t_star) / k

        w[t + 1] = w[t] - np.divide((ita_t * gradient), np.sqrt(s[t]))
        w_eDis = np.dot(w[t + 1], w[t + 1].T)
        if w_eDis > 1 / lamda:
            w[t + 1] = w[t + 1] / ((np.dot(np.multiply(np.sqrt(s[t]), w[t + 1]),
                                           np.multiply(np.sqrt(s[t].T), w[t + 1].T)) * lamda) ** 0.5)
        s[t + 1] = s[t] + np.square(gradient)

    return w
