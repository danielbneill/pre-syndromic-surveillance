""" Static Model (Classic LDA):
StaticModel is a class for background topic model training. It uses the classic LDA.
We name the parameters in mostly the same way as Griffiths et al. 2004. To learn more about LDA, please check:
https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=

n_t: Number of static topics (Also known as K in other files)
n_w: Number of words
n_d: Number of documents

Theta ~ Dirichlet(alpha), document-topic distribution, shape (n_d,n_t)
Phi ~ Dirichlet(beta), topic-word distribution, shape (n_t,n_w)

X: corpus in bag of words (BOW) format. For the sake of space, DynamicModel does not save X as a parameter.
   Please make sure you already converted X to a BOW format, [n_d, _, 2] shaped numpy array before feeding
   to LDA.
Z: word-topic assignment. Works like this: if X is [(0,3),(1,2)], meaning having three 0's and two 1's,
   then Z can be [(1,2),(0,2)], meaning there are one 0 in topic 1, two 0's in topic 2; zero 1 in topic 1,
   and two 1's in topic 2. To make it compatible with numba, we converted it to a [n_d, max-doc-len, n_t]
   shaped numpy array. We put -1 on all out-of-doc entries.

The StaticModel object -DOES NOT- save the corpus X, since it is often too big. However, it needs to take X
for initialization, as we need that to initialize Z.
"""

import numpy as np
import time
from utils import Z_to_numpy
from numba import njit

class StaticModel:
    def __init__(self, model_type):
        self.model_type=model_type

    def get_model_type(self):
        return self.static_model_type

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

    def get_topic_count(self):
        return self.topic_count

    def print_parameters(self):
        print(f"n_t (number of topics): {self.n_t}")
        print(f"n_d (number of documents): {self.n_d}")
        print(f"n_w (dictionary volume): {self.n_w}")
        print(f"alpha (Dirichlet prior for Theta, document-topic distribution): {self.alpha}")
        print(f"beta (Dirichlet prior for Phi, topic-word distribution): {self.beta}")



class StaticLDA(StaticModel):
    # Blank model initialization
    def __init__(self, X, n_d, n_w, n_t, alpha, beta):
        super().__init__('StaticLDA')
        self.n_d = n_d
        self.n_w = n_w
        self.n_t = n_t
        self.alpha = alpha
        self.beta = beta

        # --- Initialize Z at uniform distribution over each topic ---
        z = []
        max_len = 0
        for doc in X:  # document index
            z_doc = []
            for word in doc:  # word index
                count = word[1]
                if count==-1:
                    break
                avg = 1/n_t
                assigned_topic = np.random.multinomial(count, [avg] * n_t)
                z_doc.append(tuple(assigned_topic))
            if len(z_doc) > max_len:
                max_len = len(z_doc)
            z.append(z_doc)
        self.Z = Z_to_numpy(z, max_len, n_t)

        # --- Initialize Theta and Phi based on Z ---
        Theta_count = np.zeros((n_d,n_t))
        Phi_count = np.zeros((n_t,n_w))
        topic_count = np.zeros(n_t)      # Counts how many words are assigned to each topic
        for i in range(n_d):   # Document index
            for j in range(len(X[i])):  # word index
                w = X[i][j][0]
                # count = X[i][j][1]
                if w==-1:
                    break
                Z_ij = self.Z[i][j]   # This is a 1d array of length n_t, assigning the word's topic
                for k in range(n_t):
                    Theta_count[i][k] += Z_ij[k]
                    Phi_count[k][w] += Z_ij[k]
                    topic_count[k] += Z_ij[k]

        # Note that Theta and Phi here are COUNTS, not distributions.
        self.Theta = Theta_count
        self.Phi = Phi_count
        self.topic_count = topic_count

    # The train_step is a static function down below
    def train(self, X, iterations=1000, verbose=1):
        start = time.time()
        for it in range(1, iterations+1):
            self.Theta, self.Phi, self.topic_count, self.Z = train_step_staticLDA(
                self.n_d, self.n_t, self.n_w, self.Theta, self.Phi, self.topic_count,
                X, self.Z, self.alpha, self.beta)
            if verbose > 0 and it % verbose == 0:
                print(f"Iteration {it} done. Took {time.time() - start} seconds.")
                start = time.time()
        return



# The training step is made static because numba (almost) only works on static functions.
@njit
def train_step_staticLDA(n_d, n_t, n_w, Theta, Phi, topic_count, X, Z, alpha, beta):
    # In this function, Theta and Phi are counts rather than distributions. Their values
    # are integers.
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
                Phi[k][w] -= c_z[k]
                topic_count[k] -= c_z[k]
            # Calculate the topic distribution of that word based on other entries
            prob = ((Theta[i] + alpha) / (doc_len - 1 + n_t * alpha)) * \
                   ((Phi[:, w] + beta) / (topic_count + n_w * beta))
            # Sample from the distribution and plug it in
            n_z = np.random.multinomial(count, prob / sum(prob))
            Z[i][j] = n_z
            c_z = Z[i][j]
            for k in range(n_t):
                Theta[i][k] += c_z[k]
                Phi[k, w] += c_z[k]
                topic_count[k] += c_z[k]
    return Theta, Phi, topic_count, Z

