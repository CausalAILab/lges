import numpy as np
from ges.ges.scores.decomposable_score import DecomposableScore
from collections import Counter
from src.scm.ctm import CTM

class DiscreteObsL0Pen(DecomposableScore):
    """
    Implements a cached BIC score for discrete (multinomial) data.

    Assumes that all variables take discrete values from {0, 1, ..., k_i-1}
    for some number of categories k_i (can differ per variable).
    """

    def __init__(self, data, lmbda=None, cache=True, debug=0):
        """
        Parameters
        ----------
        data : numpy.ndarray
            The (n_samples, n_variables) matrix of discrete observations.
        lmbda : float or NoneType, optional
            The regularization parameter. If None, defaults to BIC, i.e., 0.5 * log(n_samples).
        cache : bool, optional
            Whether to cache local scores. Defaults to True.
        debug : int, optional
            If larger than 0, print debug information.
        """
        if type(data) != np.ndarray:
            raise TypeError("data should be a numpy.ndarray, not %s." % type(data))

        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError("data should contain integers (categorical values).")

        super().__init__(data, cache=cache, debug=debug)

        self.n, self.p = data.shape
        self.lmbda = 0.5 * np.log(self.n) if lmbda is None else lmbda

        # Number of categories for each variable
        self.cardinalities = np.array([len(np.unique(data[:, j])) for j in range(self.p)])

    def full_score(self, A):
        """
        Given a DAG adjacency A, return the full BIC score.
        """
        score = 0
        for x in range(self.p):
            pa = set(np.where(A[:, x] != 0)[0])
            score += self.local_score(x, pa)
        return score

    def _compute_local_score(self, x, pa):
        """
        Compute the local BIC score of a node x given its parents.

        Parameters
        ----------
        x : int
            A node index.
        pa : set of ints
            A set of parent node indices.

        Returns
        -------
        score : float
            The local BIC score.
        """
        data = self._data
        n = self.n

        if len(pa) == 0:
            # No parents: just the marginal distribution
            counts = Counter(data[:, x])
            log_lik = 0
            for count in counts.values():
                log_lik += count * np.log(count / n)
            num_params = self.cardinalities[x] - 1  # (k - 1) independent parameters
        else:
            # Parents: conditional distribution
            pa = list(pa)
            joint = np.column_stack([data[:, j] for j in pa] + [data[:, x]])
            counts_joint = Counter(map(tuple, joint))
            counts_pa = Counter(map(tuple, joint[:, :-1]))
            log_lik = 0
            for joint_vals, count in counts_joint.items():
                pa_vals = joint_vals[:-1]
                cond_prob = count / counts_pa[pa_vals]
                log_lik += count * np.log(cond_prob)

            num_parent_configs = np.prod(self.cardinalities[pa])
            num_params = (self.cardinalities[x] - 1) * num_parent_configs

        bic_penalty = self.lmbda * num_params
        return log_lik - bic_penalty

if __name__ == "__main__":

    np.random.seed(0)


    # Define a DAG adjacency matrix
    # Example: Variable 0 -> Variable 2, Variable 1 -> Variable 2
    A = np.array([
        [0, 0, 0],  # Variable 0 points to Variable 2
        [0, 0, 0],  # Variable 1 points to Variable 2
        [0, 0, 0]   # Variable 2 has no outgoing edges
    ])

    # Define another DAG adjacency matrix
    # Example: Variable 0 -> Variable 1, Variable 1 -> Variable 2
    B = np.array([
        [0, 1, 1],  # Variable 0 points to Variable 1 and 2
        [0, 0, 0],  # Variable 1 points to Variable 2
        [0, 1, 0]   # Variable 2 has no outgoing edges
    ])

    ctm = CTM(B)
    data = ctm.sample(1000)
    print("Generating data from B")
    score = DiscreteObsL0Pen(data, lmbda=0.5)
    print("Score for B: ", score.full_score(B))
    print("Score for A: ", score.full_score(A))