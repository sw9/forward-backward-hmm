import pickle
import math

def load_corpus(path):
    ret = []
    with open(path) as f:
        for line in f:
            for c in line:
                if c.isalpha():
                    ret.append(c.lower())

                if c.isspace() and ret and not ret[-1].isspace():
                    ret.append(" ")

    ret = "".join(ret)
    return ret.strip()


def load_probabilities(path):
    ret = None
    with open(path) as f:
        ret = pickle.load(f)
    return ret


class HMM(object):

    def __init__(self, probabilities):
        self.pi = probabilities[0]
        self.a = probabilities[1]
        self.b = probabilities[2]

    @staticmethod
    def log_sum_exp(seq):
        if abs(min(seq)) > abs(max(seq)):
            a = min(seq)
        else:
            a = max(seq)

        total = 0
        for x in seq:
            total += math.exp(x - a)
        return a + math.log(total)

    def forward(self, sequence):
        N = len(self.pi)
        alpha = []

        d1 = {}
        for i in xrange(1, N+1):
            d1[i] = self.pi[i] + self.b[i][sequence[0]]
        alpha.append(d1)

        for t in xrange(1, len(sequence)):
            d = {}
            o = sequence[t]

            for j in xrange(1, N+1):
                sum_seq = []
                for i in xrange(1, N+1):
                    sum_seq.append(alpha[-1][i] + self.a[i][j])

                d[j] = self.log_sum_exp(sum_seq) + self.b[j][o]

            alpha.append(d)

        return alpha

    def forward_probability(self, alpha):
        return self.log_sum_exp(alpha[-1].values())

    def backward(self, sequence):
        N = len(self.pi)
        T = len(sequence)
        beta = []

        dT = {}
        for i in xrange(1, N+1):
            dT[i] = 0
        beta.append(dT)

        for t in xrange(T - 2, -1, -1):
            d = {}
            o = sequence[t + 1]

            for i in xrange(1, N+1):
                sum_seq = []
                for j in xrange(1, N+1):
                    sum_seq.append(self.a[i][j] + self.b[j][o] + beta[-1][j])
                d[i] = self.log_sum_exp(sum_seq)

            beta.append(d)
        beta.reverse()
        return beta

    def backward_probability(self, beta, sequence):
        N = len(self.pi)
        sum_seq = []

        for i in xrange(1, N+1):
            sum_seq.append(self.pi[i] + self.b[i][sequence[0]] + beta[0][i])
        return self.log_sum_exp(sum_seq)

    def forward_backward(self, sequence):
        N = len(self.pi)
        alpha = self.forward(sequence)
        beta = self.backward(sequence)
        T = len(sequence)

        xis = []
        for t in xrange(T - 1):
            xis.append(self.xi_matrix(t, sequence, alpha, beta))
        gammas = []
        for t in xrange(T):
            gammas.append(self.gamma(t, sequence, alpha, beta, xis))

        pi_hat = gammas[0]
        a_hat = {}
        b_hat = {}
        for i in xrange(1, N+1):
            a_hat[i] = {}
            b_hat[i] = {}

            sum_seq = []
            for t in xrange(T - 1):
                sum_seq.append(gammas[t][i])
            a_hat_denom = self.log_sum_exp(sum_seq)

            for j in xrange(1, N+1):
                sum_seq = []
                for t in xrange(T - 1):
                    sum_seq.append(xis[t][i][j])
                a_hat_num = self.log_sum_exp(sum_seq)
                a_hat[i][j] = a_hat_num - a_hat_denom

            sum_seq = []
            for t in xrange(T):
                sum_seq.append(gammas[t][i])
            b_hat_denom = self.log_sum_exp(sum_seq)
            for k in self.b[i]:
                sum_seq = []
                for t in xrange(T):
                    o = sequence[t]
                    if o == k:
                        sum_seq.append(gammas[t][i])
                b_hat_num = self.log_sum_exp(sum_seq)
                b_hat[i][k] = b_hat_num - b_hat_denom

        return (pi_hat, a_hat, b_hat)

    def gamma(self, t, sequence, alpha, beta, xis):
        N = len(self.pi)
        gamma = {}
        if t < len(sequence) - 1:
            xi = xis[t]
            for i in xrange(1, N+1):
                sum_seq = []
                for j in xrange(1, N+1):
                    sum_seq.append(xi[i][j])
                gamma[i] = self.log_sum_exp(sum_seq)
        else:
            sum_seq = []
            for i in xrange(1, N+1):
                gamma[i] = alpha[t][i] + beta[t][i]
                sum_seq.append(gamma[i])

            denom = self.log_sum_exp(sum_seq)

            for i in xrange(1, N+1):
                gamma[i] -= denom

        return gamma

    def xi_matrix(self, t, sequence, alpha, beta):
        N = len(self.pi)
        o = sequence[t+1]

        xi = {}

        sum_seq = []

        for i in xrange(1, N+1):
            xi[i] = {}
            for j in xrange(1, N+1):
                num = alpha[t][i] + self.a[i][j] \
                      + self.b[j][o] + beta[t + 1][j]
                sum_seq.append(num)
                xi[i][j] = num

        denom = self.log_sum_exp(sum_seq)

        for i in xrange(1, N+1):
            for j in xrange(1, N+1):
                xi[i][j] -= denom

        return xi

    def update(self, sequence, cutoff_value):
        increase = cutoff_value + 1
        while (increase > cutoff_value):
            before = self.forward_probability(self.forward(sequence))
            new_p = self.forward_backward(sequence)
            self.pi = new_p[0]
            self.a = new_p[1]
            self.b = new_p[2]
            after = self.forward_probability(self.forward(sequence))
            increase = after - before
