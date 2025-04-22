import ges.ges as ges
import ges.ges.scores as scores
import ges.ges.utils as utils
from ges.ges import CIOracle
import sempler
import numpy as np
import networkx as nx
from causallearn.utils.cit import CIT

# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)
ci_oracle = ges.CIOracle(A)
ci_fisher = CIT(data, method="fisherz")

A0 = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])

# Define the score class
score_class = scores.GaussObsL0Pen(data)

cpA = utils.dag_to_cpdag(A)
# Run GES with the Gaussian BIC score
print("Without CIs")
estimate, score, metrics = ges.fit(score_class)
print(estimate, score)
# Compute the edit distance between A and estimate
edit_distance = np.sum(cpA != estimate)
print("Edit distance:", edit_distance)
print("False positives:", np.sum((cpA == 0) & (estimate != 0)))
print("False negatives:", np.sum((cpA == 1) & (estimate != 1)))
print("Num inserts eval:", metrics["num_inserts_eval"])
print("Num deletes eval:", metrics["num_deletes_eval"])
print("Num turns eval:", metrics["num_turns_eval"])
print("Num score calls:", metrics["num_inserts_eval"] + metrics["num_deletes_eval"] + metrics["num_turns_eval"])
print("Num cis eval:", len(metrics["cis_eval"]))

print("With CI Oracle")
estimate, score, metrics = ges.fit(score_class,ci=ci_oracle)
print(estimate, score)
# Compute the edit distance between A and estimate
edit_distance = np.sum(cpA != estimate)
print("Edit distance:", edit_distance)
print("False positives:", np.sum((cpA == 0) & (estimate != 0)))
print("False negatives:", np.sum((cpA == 1) & (estimate != 1)))
print("Num inserts eval:", metrics["num_inserts_eval"])
print("Num deletes eval:", metrics["num_deletes_eval"])
print("Num turns eval:", metrics["num_turns_eval"])
print("Num score calls:", metrics["num_inserts_eval"] + metrics["num_deletes_eval"] + metrics["num_turns_eval"])
print("Num cis eval:", len(metrics["cis_eval"]))


print("With CI Fisher")
estimate, score, metrics = ges.fit(score_class,ci=ci_fisher)
print(estimate, score)
# Compute the edit distance between A and estimate
edit_distance = np.sum(cpA != estimate)
print("Edit distance:", edit_distance)
print("False positives:", np.sum((cpA == 0) & (estimate != 0)))
print("False negatives:", np.sum((cpA == 1) & (estimate != 1)))
print("Num inserts eval:", metrics["num_inserts_eval"])
print("Num deletes eval:", metrics["num_deletes_eval"])
print("Num turns eval:", metrics["num_turns_eval"])
print("Num score calls:", metrics["num_inserts_eval"] + metrics["num_deletes_eval"] + metrics["num_turns_eval"])
print("Num cis eval:", len(metrics["cis_eval"]))

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 24002.112921580803