import networkx as nx
import pandas as pd
import hvplot.networkx as hvnx
import panel as pn
import pylab as plt
from holoviews import opts
from holoviews.element.graphs import layout_nodes
import matplotlib as mpl

import numpy as np
import pandas as pd
import networkx as nx
import pickle

# Sample network file paths (you can replace these with your actual file paths)
network_files = {
    'Proximity': 'proximity.graphml',
    'Survey': 'survey.graphml',
    'Facebook': 'facebook.graphml',
    'Diaries': 'diaries.graphml'
}


def load_network(file_path):
    # Load network data from file
    return nx.read_graphml(path)

def compute_centrality_measures(G, measure):
    # Calculate centrality measures
    measures = {
        'Degree': nx.degree_centrality,
        'Betweenness': nx.betweenness_centrality,
        'Closeness': nx.closeness_centrality,
        'PageRank': nx.pagerank,
        'Eigenvector': nx.eigenvector_centrality,
        # Add more centrality measures as needed
    }
    return measures[measure](G)

# Compute community detection using Louvain algorithm
def detect_communities(G):
    communities = nx.algorithms.community.louvain_communities(G, seed=42)
    partition = {node: cid for cid, comm in enumerate(communities) for node in comm}
    return partition


def display_statistics(G):
    print(G)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    diameter = nx.diameter(G)
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    num_components = nx.number_connected_components(G)
    
    data = {
        'Metric': ['Number of Nodes', 'Number of Edges', 'Density', 'Transitivity', 
                   'Assortativity', 'Diameter', 'Average Degree', 'Number of Components'],
        'Value': [num_nodes, num_edges, density, transitivity, assortativity, diameter, avg_degree, num_components]
    }
    
    df = pd.DataFrame(data)
    return pn.widgets.DataFrame(df.set_index("Metric"))

# Visualize network
def visualize_network(file_path, centrality_measure="Degree", community_measure="Community (inferred)"):
    G = nx.read_graphml(file_path)

    pos = pickle.load(open("positions_nodes.pkl", "rb"))
    
    # Compute centrality measures
    centrality = compute_centrality_measures(G, centrality_measure)
    
    # Compute community detection
    if community_measure=="Community (inferred)":
        communities = detect_communities(G)
    else:
        communities = nx.get_node_attributes(G, community_measure)

    communities = [communities[node] for node in G.nodes]
    cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Create a mapping dictionary
    mapping_dict = {com: index for index, com in enumerate(set(communities))}
    mapping_dict_r = {i: com for (com, i) in mapping_dict.items()}
    # Map letters to indexes using list comprehension
    indexes = [mapping_dict[com] for com in communities]
    color = [cs[i%10] for i in indexes]

    
    # Prepare size based on centrality
    size = np.array([centrality[node] for node in G.nodes()])
    size = (size-np.min(size))/(np.max(size)-np.min(size))
    size *= 100
    size += 10

    # Draw network
    spring = hvnx.draw(G, 
                       pos = {node: pos[int(node)] for node in G},
                       node_size=size, 
                       node_color=color, 
                       edge_color='darkgray', 
                       edge_alpha=0.5
                      )
    

    return spring
    


# Create file selector widget
file_selector = pn.widgets.Select(options=network_files, name='Select Network File')

# Create centrality measure selector widget
centrality_selector = pn.widgets.Select(options=['Degree', 'Betweenness', 'Closeness', 'PageRank', 'Eigenvector'], name='Node size: Centrality Measure')

# Create community detection toggle
community_toggle = pn.widgets.Select(options=['Community (inferred)', 'Gender', 'Program', 'Classroom'], name='Node coor: Centrality Measure')

# Create panel app layout
def update_app(file_path, centrality_measure, community_detection):
    network_plot = visualize_network(file_path, centrality_measure, community_detection)
    #stats_table = display_statistics(file_path)
    return pn.Row(
        pn.Column(file_selector, centrality_selector, community_toggle),
        pn.Column(network_plot)
    )

app = pn.bind(update_app,
        file_path=file_selector.param.value, 
        centrality_measure=centrality_selector.param.value, 
        community_detection=community_toggle.param.value)



# Display the app
layout = pn.Row(app)
layout.servable()