import torch

RANDOM_SEED = 42


class CTM:
    def __init__(self, adj_matrix, cardinalities=None, random_state=RANDOM_SEED):
        """
        Initialize the Canonical Type Model using an adjacency matrix.

        Args:
            adj_matrix (list[list[int]]): The adjacency matrix of the DAG.
                                          A[i][j] = 1 indicates a directed edge from variable V_i to variable V_j.
            cardinalities (dict): A dictionary mapping each variable index to the number of discrete values it can take.
            random_state (int): Random seed for reproducibility.
        """

        torch.manual_seed(random_state)  # Set the random seed for reproducibility

        self.adj_matrix = adj_matrix
        if cardinalities is None:
            # Default cardinalities: assume binary variables
            self.cardinalities = {i: 2 for i in range(len(adj_matrix))}
        else:
            self.cardinalities = cardinalities
        self.variables = list(range(len(self.adj_matrix)))  # Variable indices as list of integers

        # Convert the adjacency matrix to a graph (dict of parent relationships)
        self.graph = self.adjacency_to_graph()
        self.topo_order = self.topological_sort()
        self.random_state = random_state

        # Initialize conditional probability parameters
        self.conditionals = {}
        for var in self.variables:
            parent_vars = self.graph[var]
            if not parent_vars:
                # No parents: use a marginal categorical
                self.conditionals[var] = torch.rand(1, self.cardinalities[var])
                self.conditionals[var] = self.conditionals[var] / self.conditionals[var].sum()
            else:
                parent_cardinalities = [self.cardinalities[p] for p in parent_vars]

                num_parent_configs = 1
                for c in parent_cardinalities:
                    num_parent_configs *= c

                # For each parent configuration, a categorical over var's values
                probs = torch.rand(num_parent_configs, self.cardinalities[var])
                probs = probs / probs.sum(dim=-1, keepdim=True)  # normalize
                self.conditionals[var] = probs  # shape: (num_parent_configs, cardinality[var])

    def adjacency_to_graph(self):
        """
        Convert the adjacency matrix to a dictionary of parent variables.

        Returns:
            dict: A dictionary mapping each variable to its parent variables.
        """
        graph = {}
        for i in range(len(self.adj_matrix)):
            parents = [j for j in range(len(self.adj_matrix)) if self.adj_matrix[i][j] == 1]
            graph[i] = parents
        return graph
    
    def topological_sort(self):
        order = []
        visited = set()

        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for p in self.graph[v]:
                dfs(p)
            order.append(v)

        for v in self.graph:
            dfs(v)
        return order

    def parent_indices(self, parent_samples, parent_cardinalities):
        """
        Convert a batch of parent samples into an index in 0..num_parent_configs-1
        """
        idx = torch.zeros_like(parent_samples[:, 0])
        multiplier = 1
        for i in reversed(range(len(parent_cardinalities))):
            idx += parent_samples[:, i] * multiplier
            multiplier *= parent_cardinalities[i]
        return idx

    def sample(self, n, interventions={}):
        samples = {}

        for var in self.topo_order:
            if var in interventions:
                samples[var] = torch.full((n,), interventions[var])  # Set the variable to the intervention value.
            else:
                parents = self.graph[var]
                if not parents:
                    # No parents: use a marginal categorical
                    probs = self.conditionals[var][0]  # shape (cardinality,)
                    dist = torch.distributions.Categorical(probs=probs)
                    samples[var] = dist.sample((n,))
                else:
                    parent_samples = torch.stack([samples[p] for p in parents], dim=-1)
                    parent_cardinalities = [self.cardinalities[p] for p in parents]

                    parent_idx = self.parent_indices(parent_samples, parent_cardinalities)
                    probs = self.conditionals[var][parent_idx]  # shape (n, cardinality)

                    dist = torch.distributions.Categorical(probs=probs)
                    samples[var] = dist.sample()

        # Stack all the variables into a single tensor following the order
        stacked = torch.stack([samples[var] for var in self.variables], dim=1)  # shape (n, n_variables)

        return stacked.numpy()

if __name__ == '__main__':
    # Define the adjacency matrix as a normal list of lists
    adj_matrix = [
        [0, 1, 0],  # X -> Y
        [0, 0, 1],  # Y -> Z
        [0, 0, 0]   # Z has no children
    ]

    # Create the CTM model with a random seed
    ctm = CTM(adj_matrix, random_state=42)

    # Sample 10 data points
    samples = ctm.sample(10)
    print(samples)

    # Intervene on X and Y
    interventions = {'X': 1, 'Y': 2}  # X = 1, Y = 2
    samples_with_interventions = ctm.sample(10, interventions=interventions)
    print(samples_with_interventions)
