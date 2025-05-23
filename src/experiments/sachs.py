import ges.ges as ges
import ges.ges.utils as utils
from causallearn.utils.cit import CIT
import pandas as pd
import numpy as np
from src.experiments.utils import cpdag_shd
import urllib.request
import os
import time

# Fetch and preprocess observational data
def fetch_and_preprocess_data(url):
    # Use pooch to robustly download the data
    # Define the file path
    file_path = "/tmp/sachs_discretised.txt.gz"

    # Download using urllib if file doesn't exist
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
    # Read the data
    df = pd.read_csv(file_path, delimiter=" ")
    df_mapped = df.copy()

    # Map the columns to integers from intervals
    for col in df.columns:
        unique_bins = sorted(df[col].unique(), key=lambda x: float(x.split(',')[0].strip('(')))
        bin_mapping = {bin_label: idx for idx, bin_label in enumerate(unique_bins)}
        df_mapped[col] = df[col].map(bin_mapping)

    return df_mapped

def print_mec(mec):
    for i in range(mec.shape[0]):
        for j in range(mec.shape[1]):
            if mec[i, j] == 1 and mec[j, i] == 0:
                src = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
                dst = list(node_mapping.keys())[list(node_mapping.values()).index(j)]
                print(f"Edge: {src} -> {dst}")
            elif mec[j, i] == 1 and mec[i, j] == 1 and i < j:
                src = list(node_mapping.keys())[list(node_mapping.values()).index(i)]
                dst = list(node_mapping.keys())[list(node_mapping.values()).index(j)]
                print(f"Edge: {src} - {dst}")

# Define the Sachs graph structure
def define_sachs_graph():
    node_mapping = {
        "Raf": 0, "Mek": 1, "Plcg": 2, "PIP2": 3, "PIP3": 4,
        "Erk": 5, "Akt": 6, "PKA": 7, "PKC": 8, "P38": 9, "Jnk": 10,
    }
    num_nodes = len(node_mapping)
    sachs_graph = np.zeros((num_nodes, num_nodes), dtype=int)

    # Define directed edges
    sachs_directed_edges = [
        ("Erk", "Akt"), ("PKA", "Akt"), ("PKA", "Erk"), ("PKA", "Jnk"),
        ("PKA", "Mek"), ("PKA", "P38"), ("PKA", "Raf"), ("Mek", "Erk"),
        ("PKC", "Jnk"), ("PKC", "Mek"), ("PKC", "P38"), ("PKC", "PKA"),
        ("PKC", "Raf"), ("Raf", "Mek"), ("PIP3", "PIP2"), ("Plcg", "PIP2"),
        ("Plcg", "PIP3"),
    ]

    # Populate adjacency matrix
    for src, dst in sachs_directed_edges:
        sachs_graph[node_mapping[src], node_mapping[dst]] = 1

    return sachs_graph, node_mapping

# Define supplemental graph with low-confidence edges
def define_supplemental_graph():
    sachs_supplemental_graph, node_mapping = define_sachs_graph()

    # Define undirected edges
    sachs_undirected_edges = [
        ("PKC", "Akt"), ("Raf", "Akt"), ("Mek", "Akt"), ("Akt", "Plcg"),
        ("Mek", "Plcg"), ("Mek", "Jnk"), ("PKA", "Plcg"), ("Jnk", "P38"),
    ]

    # Populate adjacency matrix
    for src, dst in sachs_undirected_edges:
        sachs_supplemental_graph[node_mapping[src], node_mapping[dst]] = 1
        sachs_supplemental_graph[node_mapping[dst], node_mapping[src]] = 1

    return utils.pdag_to_dag(sachs_supplemental_graph)


# Main execution
if __name__ == "__main__":
   

    # Define Sachs graph
    sachs_graph, node_mapping = define_sachs_graph()
    sachs_mec = utils.dag_to_cpdag(sachs_graph)

    # Experiment with discretized data
    url = "https://www.bnlearn.com/book-crc/code/sachs.discretised.txt.gz"
    df = fetch_and_preprocess_data(url)
    data = df.to_numpy()

    score_class = ges.scores.DiscreteObsL0Pen(data)

    print("Discretized Data")

    # run GES
    start = time.time()
    estimate_ges, metrics = ges.fit(score_class)
    end = time.time()
    print("GES SHD:", cpdag_shd(sachs_mec, estimate_ges))
    print("GES Time:", end - start)

    # run LGES with ConservativeInsert
    start = time.time()
    estimate_lges_cons, metrics = ges.fit(score_class,
                                          score_based=True,
                                          prune=True)
    end = time.time()
    print("LGES (Cons) SHD:", cpdag_shd(sachs_mec, estimate_lges_cons))
    print("LGES (Cons) Time:", end - start)

    # run LGES with SafeInsert
    start = time.time()
    estimate_lges_safe, metrics = ges.fit(score_class,
                                          score_based=True,
                                          prune=False)
    end = time.time()
    print("LGES (Safe) SHD:", cpdag_shd(sachs_mec, estimate_lges_safe))
    print("LGES (Safe) Time:", end - start)

    # Check if SHD between GES and LGES estimates is 0
    shd_ges_lges_cons = cpdag_shd(estimate_ges, estimate_lges_cons)
    shd_ges_lges_safe = cpdag_shd(estimate_ges, estimate_lges_safe)
    shd_lges_lges = cpdag_shd(estimate_lges_cons, estimate_lges_safe)
    print("SHD between GES and LGES (Cons):", shd_ges_lges_cons)
    print("SHD between GES and LGES (Safe):", shd_ges_lges_safe)
    print("SHD between LGES (Cons) and LGES (Safe):", shd_lges_lges)

    
    # Print the true DAG
    print("True MEC (Sachs Graph):")
    print_mec(sachs_mec)
    
    # Print the GES estimate
    print("GES Estimated MEC:")
    print_mec(estimate_ges)


    # Experiment with continuous data
    print("Continuous Data")

    # Read the file into a numpy array
    file_path = "./sachs_data.txt"

    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the file into a pandas DataFrame and convert to numpy array
    df = pd.read_csv(file_path, delimiter="\t")
    data = df.to_numpy()
    

    score_class = ges.scores.GaussObsL0Pen(data)

    # run GES
    start = time.time()
    estimate_ges, metrics = ges.fit(score_class)
    end = time.time()
    print("GES SHD:", cpdag_shd(sachs_mec, estimate_ges))
    print("GES Time:", end - start)

    # run LGES with ConservativeInsert
    start = time.time()
    estimate_lges_cons, metrics = ges.fit(score_class,
                        score_based=True,
                        prune=True)
    end = time.time()
    print("LGES (Cons) SHD:", cpdag_shd(sachs_mec, estimate_lges_cons))
    print("LGES (Cons) Time:", end - start)

    # run LGES with SafeInsert
    start = time.time()
    estimate_lges_safe, metrics = ges.fit(score_class,
                        score_based=True,
                        prune=False)
    end = time.time()
    print("LGES (Safe) SHD:", cpdag_shd(sachs_mec, estimate_lges_safe))
    print("LGES (Safe) Time:", end - start)

    # Check if SHD between GES and LGES estimates is 0
    shd_ges_lges_cons = cpdag_shd(estimate_ges, estimate_lges_cons)
    shd_ges_lges_safe = cpdag_shd(estimate_ges, estimate_lges_safe)
    shd_lges_lges = cpdag_shd(estimate_lges_cons, estimate_lges_safe)
    print("SHD between GES and LGES (Cons):", shd_ges_lges_cons)
    print("SHD between GES and LGES (Safe):", shd_ges_lges_safe)
    print("SHD between LGES (Cons) and LGES (Safe):", shd_lges_lges)

    # Print the GES estimate
    print("GES Estimated MEC:")
    print_mec(estimate_ges)
