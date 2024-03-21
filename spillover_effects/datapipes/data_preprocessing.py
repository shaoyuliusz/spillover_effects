import gzip
import json
import pandas as pd
import networkx as nx
import pickle
import os
import argparse


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def getBipartite(
    df: pd.DataFrame, node_0: str, node_1: str, weight_name=None
) -> nx.Graph:
    """
    construct bipartite graph from a pandas dataframe
    Args:
        df: pandas dataframe
        node_0: name for node 0 in bipartite graph
        node_1: name for node 1 in bipartite graph
        weight_name: if edge are weighted
    Returns:
        nx.Graph object
    """
    if node_0 not in df.columns:
        raise ValueError("node 0 is not in column names.")
    if node_1 not in df.columns:
        raise ValueError("node 1 is not in column names.")
    if weight_name is not None and weight_name not in df.columns:
        raise ValueError("weight name is not in column names.")

    B = nx.Graph()

    if weight_name is not None:
        B.add_weighted_edges_from(
            [(row[node_0], row[node_1], 1) for idx, row in df.iterrows()],
            weight=weight_name,
        )
    else:
        B.add_edges_from([(row[node_0], row[node_1]) for idx, row in df.iterrows()])

    return B


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file and save the output.")

    # Add arguments
    parser.add_argument("--data_base_dir", type=str, help="data directory.")
    parser.add_argument("--save_dir", type=str, help="save directory.")
    parser.add_argument("--file_name", type=str, help="file name.")
    parser.add_argument("--node_0", type=str, help="node 0 name for bipartite graph.")
    parser.add_argument("--node_1", type=str, help="node 1 name for bipartite graph.")
    parser.add_argument(
        "weight_name", default=None, help="name for bipartite graph edge weights."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    data_base_dir = args.data_base_dir
    save_dir = args.save_dir
    file_name = args.file_name
    node_0_name = args.node_0
    node_1_name = args.node_1
    weight_name = args.weight_name

    # create input and output paths
    input_path = os.path.join(data_base_dir, file_name, ".json.gz")
    output_path = os.path.join(save_dir, file_name, ".pkl")

    df = getDF(input_path)
    bi_graph = getBipartite(
        df, node_0=node_0_name, node_1=node_1_name, weight_name=weight_name
    )
    # TO ADD: save the outcome col Y here
    pickle.dump(bi_graph, open(output_path, "wb"))
