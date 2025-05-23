import numpy as np
import sempler
import ges.ges as ges
from src.experiments.utils import cpdag_shd




# Create a directed acyclic graph (DAG) as an adjacency matrix
G = np.array([
    [0, 1, 0, 0],  # X1 -> Z
    [0, 0, 1, 1],  # Z -> X2, Z -> Y
    [0, 0, 0, 0],  # X2
    [0, 0, 0, 0]   # Y
])



# compare graphs
# A: X1 -> Z, Z -> Y

adj_A = np.array([
    [0, 1, 0, 0],  # X1 -> Z
    [0, 0, 0, 1],  # Z -> Y
    [0, 0, 0, 0],  # X2
    [0, 0, 0, 0]   # Y
])

# B: X1 -> Y, Z -> Y
adj_B = np.array([
    [0, 0, 0, 1],  # X1 -> Y
    [0, 0, 0, 1],  # Z -> Y
    [0, 0, 0, 0],  # X2
    [0, 0, 0, 0]   # Y
])

# C: X1 -> Y, Y -> Z
adj_C = np.array([
    [0, 0, 0, 1],  # X1 -> Y
    [0, 0, 1, 0],  # Y -> Z
    [0, 0, 0, 0],  # X2
    [0, 0, 0, 0]   # Y
])



# generate random data
count_A_higher_B = 0
count_B_higher_C = 0
count_A_higher_C = 0
count_A_highest = 0
count_B_highest = 0
count_C_highest = 0

for i in range(100):
    W = G * np.random.choice([-1, 1], size=G.shape) * np.random.uniform(0.5, 2, G.shape)
    # sample noise means from N(0, 1)
    noise_means = np.zeros(G.shape[0])
    # sample noise variances from U(0.1, 0.5)
    noise_variances = np.random.uniform(0.1, 0.5, size=G.shape[0])
    sampler = sempler.LGANM(W, noise_means, noise_variances, random_state=i)
    data = sampler.sample(n=1000)
    score_class = ges.GaussObsL0Pen(data)
    score_A = score_class.full_score(adj_A)
    score_B = score_class.full_score(adj_B)
    score_C = score_class.full_score(adj_C)

    # Compare scores
    if score_A > score_B:
        count_A_higher_B += 1
    if score_B > score_C:
        count_B_higher_C += 1
    if score_A > score_C:
        count_A_higher_C += 1

    # Count which scores the highest
    if score_A > score_B and score_A > score_C:
        count_A_highest += 1
    elif score_B > score_A and score_B > score_C:
        count_B_highest += 1
    elif score_C > score_A and score_C > score_B:
        count_C_highest += 1

# Calculate fractions
fraction_A_higher_B = count_A_higher_B / 100
fraction_B_higher_C = count_B_higher_C / 100
fraction_A_higher_C = count_A_higher_C / 100
fraction_A_highest = count_A_highest / 100
fraction_B_highest = count_B_highest / 100
fraction_C_highest = count_C_highest / 100

# Print results
print("Fraction of times A scores higher than B:", fraction_A_higher_B)
print("Fraction of times B scores higher than C:", fraction_B_higher_C)
print("Fraction of times A scores higher than C:", fraction_A_higher_C)
print("Fraction of times A scores the highest:", fraction_A_highest)
print("Fraction of times B scores the highest:", fraction_B_highest)
print("Fraction of times C scores the highest:", fraction_C_highest)

