import numpy as np
import networkx as nx
from typing import Union


class Dataset:
    def __init__(
        self, graph: nx.Graph, edge_weight_attr: Union[None, str] = "weight"
    ) -> None:
        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "data must be of nx.Graph type. "
                f"{str(graph)} of type {str(type(graph))} was passed."
            )
        self.graph = graph
        self.edge_weight_attr = edge_weight_attr
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        self.adj_matrix = nx.adjacency_matrix(self.graph)

    def __str__(self):
        data_summary = self._data_summary_str()
        degree_summary = self._degree_summary()
        edge_summary = self._edge_summary()

        res = (
            "================== DataSet Object ==================\n"
            + "\n------------------ Data summary ------------------\n"
            + data_summary
            + degree_summary
            + edge_summary
        )
        return res

    def _data_summary_str(self):
        data_summary = f"No. Nodes: {self.num_nodes}\n" f"No. Edges: {self.num_edges}\n"
        return data_summary

    def _degree_summary(self):
        # Compute node degree summary statistics
        node_degrees = [d for n, d in self.graph.degree()]
        node_degree_summary = {
            "min": np.min(node_degrees),
            "max": np.max(node_degrees),
            "mean": np.mean(node_degrees),
            "median": np.median(node_degrees),
        }

        summary_str = (
            f"Node degree summary: \n"
            f" - Min: {node_degree_summary['min']:.2f}\n"
            f" - Max: {node_degree_summary['max']:.2f}\n"
            f" - Mean: {node_degree_summary['mean']:.2f}\n"
            f" - Median: {node_degree_summary['median']:.2f}\n"
        )
        return summary_str

    def _edge_summary(self):
        # Compute edge weights summary statistics
        if not self.edge_weight_attr:
            return f"Edge summary: \n" f" - Edge is unweighted"

        edge_weights = [
            d[self.edge_weight_attr] for u, v, d in self.graph.edges(data=True)
        ]
        edge_weight_summary = {
            "min": np.min(edge_weights),
            "max": np.max(edge_weights),
            "mean": np.mean(edge_weights),
            "median": np.median(edge_weights),
        }

        summary_str = (
            f"Edge summary: \n"
            f" - Min: {edge_weight_summary['min']:.2f}\n"
            f" - Max: {edge_weight_summary['max']:.2f}\n"
            f" - Mean: {edge_weight_summary['mean']:.2f}\n"
            f" - Median: {edge_weight_summary['median']:.2f}"
        )
        return summary_str


class BipartiteDataset(Dataset):
    """dataset for bipartite graph data

    Parameters
    ----------
    graph : :class:`nx.Graph` object

    edge_weight_attr : graph attribute specifying edge weights, default is None

    bi_attr_key : attribute key specifying bipartite node types, default is "bipartite"

    bi_attr_outcome: attribute value for outcome units, default is 0

    bi_attr_diversion: attribute value for diversion units, default is 1

    """

    def __init__(
        self,
        graph,
        edge_weight_attr: str = "weight",
        bi_attr_key: str = "bipartite",
        bi_attr_outcome=0,
        bi_attr_diversion=1,
    ):
        super().__init__(graph, edge_weight_attr)
        self.bi_attr_key = bi_attr_key
        self.bi_attr_outcome = bi_attr_outcome
        self.bi_attr_diversion = bi_attr_diversion

        n_outcome, n_diversion = self._count_units()
        self.n_outcome = n_outcome
        self.n_diversion = n_diversion

    def _data_summary_str(self):
        data_summary = (
            f"No. Nodes: {self.num_nodes}\n"
            f"No. Edges: {self.num_edges}\n"
            f"No. Outcome nodes: {self.n_outcome}\n"
            f"No. Diversion nodes: {self.n_diversion}\n"
        )
        return data_summary

    def _compute_node_degrees(self, node_type) -> np.ndarray:
        """warnings we assume node order do not change"""
        if node_type == "outcome":
            nodes = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get(self.bi_attr_key) == self.bi_attr_outcome
            ]
        elif node_type == "diversion":
            nodes = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get(self.bi_attr_key) == self.bi_attr_diversion
            ]
            # Compute the degree for each food node
        return np.array([self.graph.degree(node) for node in nodes])

    @property
    def degree_outcome(self):
        return self._compute_node_degrees(node_type="outcome")

    @property
    def degree_diversion(self):
        return self._compute_node_degrees(node_type="diversion")

    def _count_units(self):
        """Count number of outcome and diversion units"""
        outcome_count = sum(
            1
            for node, data in self.graph.nodes(data=True)
            if self.bi_attr_key in data
            and data[self.bi_attr_key] == self.bi_attr_outcome
        )
        n_outcome = outcome_count
        n_diversion = self.graph.number_of_nodes() - outcome_count
        return n_outcome, n_diversion
