import numpy as np

class HMM:
    def __init__(self, observed, transition_matrix, emission_matrix, initial_distribution):
        self.I = initial_distribution
        self.V = np.array(observed)
        self.A = np.array(transition_matrix)
        self.B = np.array(emission_matrix)

        self.K = self.A.shape[0]
        self.N = self.V.shape[0]


    def forward(self):
        alpha = np.zeros((self.N, self.K))

        alpha[0, :] = self.I * self.B[:, self.V[0]]
        for t in range(1, self.N):
            for j in range(self.K):
                for i in range(self.K):
                    alpha[t, j] += alpha[t - 1, i] * self.A[i, j] * self.B[j, self.V[t]]

        return np.argmax(alpha, axis=1), alpha

    def backward(self):
        beta = np.zeros((self.N, self.K))

        beta[self.N - 1, :] = np.ones(self.K)
        for t in range(self.N - 2, -1, -1):
            for j in range(self.K):
                for i in range(self.K):
                    beta[t, j] += beta[t + 1, i] * self.A[j, i] * self.B[i, self.V[t + 1]]

        return np.argmax(beta, axis=1), beta

    def forward_backward(self):
        fbv = np.zeros((self.N, self.K))
        # TODO: calculate forward-backward values
        alfa_max, alfa = self.forward()
        beta_max, beta = self.backward()
        fbv = alfa * beta

        return np.argmax(fbv, axis=1)

    def viterbi(self):
        T1 = self.V.shape[0]
        T2 = self.A.shape[0]
        omega = np.zeros((T1, T2))
        omega[0, :] = np.log(self.I * self.B[:, self.V[0]])
        prev = np.zeros((T1 - 1, T2))

        for t in range(1, T1):
            for j in range(T2):
                np.seterr(divide='ignore')
                probability = omega[t - 1] + np.log(self.A[:, j]) + np.log(self.B[j, self.V[t]])
                np.seterr(divide='warn')
                prev[t - 1, j] = np.argmax(probability)
                omega[t, j] = np.max(probability)
        S = np.zeros(T1)
        last_state = np.argmax(omega[T1 - 1, :])

        S[0] = last_state

        backtrack_index = 1
        for i in range(T1 - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        vitrebi = np.flip(S, axis=0)

        return vitrebi
