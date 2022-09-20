""" Contrastive Model:
ContrastiveModel is a class for foreground topic model training. It is called in SpatialScan.
Symbols for most parameters follow Griffiths et al. 2004:
https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=

K: Number of static topics
K_prime: Number of foreground topics
n_w: Number of words
n_d: Number of documents

Theta ~ Dirichlet(alpha), document-topic distribution, shape (n_d, K+K_Prime)
Phi ~ Dirichlet(beta), topic-word distribution, shape (K+K_Prime,n_w)

X: corpus in bag of words (BOW) format. For the sake of space, DynamicModel does not save X as a parameter.
   Please make sure you already converted X to a BOW format, [n_d, _, 2] shaped numpy array before feeding
   to LDA.
Z: word-topic assignment. Works like this: if X is [(0,3),(1,2)], meaning having three 0's and two 1's,
   then Z can be [(1,2),(0,2)], meaning there are one 0 in topic 1, two 0's in topic 2; zero 1 in topic 1,
   and two 1's in topic 2. To make it compatible with numba, we converted it to a [n_d, max-len, 2] shaped
   numpy array.

The DynamicModel -DOES NOT- save the corpus X, since it is often too big. However, it does take the corpus for
initialization, as we need that to initialize Z.
"""
import numpy as np
from numba import njit
import time
from utils import Z_to_numpy, wt_from_Phi


class ContrastiveModel:
    def __init__(self, dynamic_model_type):
        self.model_type = dynamic_model_type

    def get_model_type(self):
        return self.model_type

    def get_n_d(self):
        return self.n_d

    def get_n_t(self):
        return self.n_t

    def get_n_w(self):
        return self.n_w

    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def get_Theta(self):
        return self.Theta

    def get_Phi(self):
        return self.Phi

    def get_Z(self):
        return self.Z

    def print_parameters(self):
        print(f"n_t (number of topics): {self.n_t}")
        print(f"n_d (number of documents): {self.n_d}")
        print(f"n_w (dictionary volume): {self.n_w}")
        print(f"alpha (Dirichlet prior for Theta, document-topic distribution): {self.alpha}")
        print(f"beta (Dirichlet prior for Phi, topic-word distribution): {self.beta}")


class ContrastiveLDA(ContrastiveModel):
    # You should have already trained statically, K topics on background data,
    # and K' topics on foreground data.
    # This step aims at making the foreground "contrastive" to the background.
    # Phi_b should have shape [K,  n_w]
    # Phi_f should have shape [K', n_w]
    def __init__(self, X, Phi_b, Phi_f, alpha, beta):
        super().__init__('ContrastiveLDA')
        self.static_Phi = np.array(Phi_b)
        self.n_w = Phi_b.shape[1]
        self.n_d = len(X)
        self.K = Phi_b.shape[0]
        self.K_prime = Phi_f.shape[0]
        self.n_t = self.K + self.K_prime
        self.alpha = alpha
        self.beta = beta
        Phi_dist = np.concatenate((Phi_b, Phi_f))

        # Initialize foreground Z (Z_f) with Phi_dist
        Z = []
        max_len = 0
        for doc in X:  # loop through the foreground docs only
            Z_doc = []
            for word in doc:  # word index
                w = word[0]
                if w==-1:
                    break
                count = word[1]
                wt_dist = wt_from_Phi(Phi_dist, w)
                assigned_topic = np.random.multinomial(count, wt_dist)
                Z_doc.append(assigned_topic.tolist())
            Z.append(Z_doc)
            if len(Z_doc) > max_len:
                max_len = len(Z_doc)
        Z = Z_to_numpy(Z, max_len, self.n_t)
        self.Z = Z

        # Create a count-valued new Theta and Phi based on Z. Note here n_d and n_t are
        # both foreground + background
        Theta = np.zeros((self.n_d, self.n_t))
        Phi = np.zeros((self.n_t, self.n_w))
        topic_count = np.zeros(self.n_t)
        for i in range(self.n_d):   # document index
            for j in range(len(X[i])):  # word index
                w = X[i][j][0]
                if w==-1:
                    break
                c_z = Z[i][j]
                for k in range(self.n_t):
                    Theta[i][k] += c_z[k]
                    topic_count[k] += c_z[k]
                    Phi[k][w] += c_z[k]
        self.Theta = Theta
        self.Phi = Phi
        self.topic_count = topic_count


    def train(self, X, iterations=2000, verbose=-1):
        start = time.time()
        for it in range(1, iterations + 1):
            self.Theta, self.Phi, self.topic_count, self.Z = train_step_ContrastiveLDA(self.n_d,
                self.n_t, self.n_w, self.K, np.array(self.Theta), np.array(self.Phi), X, np.array(self.Z),
                np.array(self.topic_count), self.alpha, self.beta)
            if verbose > 0 and it % verbose == 0:
                print(f"Iteration {it} done. Took {time.time() - start} seconds.")
                start = time.time()
        return


@njit
def train_step_ContrastiveLDA(n_d, n_t, n_w, K, Theta, Phi, X, Z, topic_count, alpha, beta):
    for i in range(n_d):
        doc_len = sum([j[1] for j in X[i] if j[1]>0])
        for j in range(len(X[i])):
            # Spot the current word to take away and re-assign
            w = X[i][j][0]
            if w==-1:
                break
            count = X[i][j][1]
            c_z = Z[i][j]
            for k in range(n_t):
                Theta[i][k] -= c_z[k]
                topic_count[k] -= c_z[k]
                # We don't change background Phi's
                if k >= K:
                    Phi[k][w] -= c_z[k]
            # Calculate the topic distribution of that word based on other entries
            prob = ((Theta[i] + alpha) / (doc_len - 1 + n_t * alpha)) * \
                   ((Phi[:, w] + beta) / (topic_count + n_w * beta))
            # Sample from the distribution and plug it in
            n_z = np.random.multinomial(count, prob / sum(prob))
            Z[i][j] = n_z
            c_z = Z[i][j]
            for k in range(n_t):
                Theta[i][k] += c_z[k]
                topic_count[k] += c_z[k]
                if k>=K:
                    Phi[k][w] += n_z[k]
    return Theta, Phi, topic_count, Z
