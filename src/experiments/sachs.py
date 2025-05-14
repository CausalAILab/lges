import ges
import ges.ges.utils as utils
from causallearn.utils.cit import CIT
import pooch
import pandas as pd
import numpy as np

# Fetch and preprocess observational data
def fetch_and_preprocess_data(url):
    # Use pooch to robustly download the data
    file_path = pooch.retrieve(url=url, known_hash=None)
    # Read the data
    df = pd.read_csv(file_path, delimiter=" ")
    df_mapped = df.copy()

    # Map the columns to integers from intervals
    for col in df.columns:
        unique_bins = sorted(df[col].unique(), key=lambda x: float(x.split(',')[0].strip('(')))
        bin_mapping = {bin_label: idx for idx, bin_label in enumerate(unique_bins)}
        df_mapped[col] = df[col].map(bin_mapping)

    return df_mapped

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

# Evaluate a graph using GES
def evaluate_graph(score_class, input_graph=None,ci=None):

    if input_graph is not None:
        score = score_class.full_score(input_graph)
    estimate, estimated_score, metrics = ges.fit(score_class, A0=input_graph, ci=ci)
    metrics["cis_eval"] = len(metrics["cis_eval"])

    if input_graph is not None:

        print("Score of input graph:", score)

    print("Score of estimated graph:", estimated_score)
    print("Metrics:", metrics)

    if input_graph is not None:
        edit_distance = np.sum(input_graph != estimate)
        print("Edit distance between input graph and estimate:", edit_distance)

# Main execution
if __name__ == "__main__":
    # Fetch and preprocess data
    url = "https://www.bnlearn.com/book-crc/code/sachs.discretised.txt.gz"
    df = fetch_and_preprocess_data(url)
    data = df.to_numpy()

    # Define Sachs graph
    sachs_graph, node_mapping = define_sachs_graph()

    # Initialize score class
    score_class = ges.scores.GaussObsL0Pen(data)

    ci_chisq = CIT(data, method="chisq")

    for ci in [None, ci_chisq]:

        print(f"\n---RUNNING WITH CI: {ci is not None}\n")
        # Evaluate without initialization   
        print("Running without initialization:")
        evaluate_graph(score_class, ci=ci)

         # Evaluate Sachs graph
        print("Running with Sachs graph as input:")
        evaluate_graph(score_class, input_graph=utils.dag_to_cpdag(sachs_graph), ci=ci)

        # Define and evaluate supplemental graph
        # sachs_supplemental_graph = define_supplemental_graph()

        # print("Running with supplemental Sachs graph as input:")
        # evaluate_graph(score_class, input_graph=utils.dag_to_cpdag(sachs_supplemental_graph), ci=ci)

