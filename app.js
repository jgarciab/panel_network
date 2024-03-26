importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.4/pyc/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/1.2.3/dist/wheels/bokeh-3.2.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.2.3/dist/wheels/panel-1.2.3-py3-none-any.whl', 'pyodide-http==0.2.1', 'holoviews', 'hvplot', 'matplotlib', 'networkx', 'numpy', 'pandas', 'pylab', 'requests', 'scipy']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

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
import requests
from io import BytesIO
import scipy

# Sample network file paths (you can replace these with your actual file paths)
network_files = {
    'Proximity': 'https://javier.science/panel_network/data/proximity.graphml',
    'Survey': 'https://javier.science/panel_network/data/survey.graphml',
    'Facebook': 'https://javier.science/panel_network/data/facebook.graphml',
    'Diaries': 'https://javier.science/panel_network/data/diaries.graphml'
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

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    gender_assort = nx.assortativity.attribute_assortativity_coefficient(G, "Gender")
    class_assort = nx.assortativity.attribute_assortativity_coefficient(G, "Classroom")
    program_assort = nx.assortativity.attribute_assortativity_coefficient(G, "Program")
    
    diameter = nx.diameter(G)
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    num_components = nx.number_connected_components(G)
    
    stats = f"Number of Nodes: {num_nodes}<br>"
    stats += f"Number of Edges: {num_edges}<br>"
    stats += f"Average Degree: {avg_degree:2.2f}<br>"
    stats += f"Diameter: {diameter}<br>"
    stats += f"Density: {density:2.2f}<br>"
    stats += f"Transitivity: {transitivity:2.2f}<br>"
    stats += f"Degree assortativity: {assortativity:2.2f}<br>"
    stats += f"Gender assortativity: {gender_assort:2.2f}<br>"
    stats += f"Clasroom assortativity: {class_assort:2.2f}<br>"
    stats += f"Program assortativity: {program_assort:2.2f}<br>"
    
    return pn.pane.HTML(stats)

# Visualize network
def visualize_network(G, centrality_measure="Degree", community_measure="Community (inferred)"):
    # Load node positions
    pos_response = requests.get("https://javier.science/panel_network/data/positions_nodes.pkl")
    pos = pickle.load(BytesIO(pos_response.content))
    
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
    response = requests.get(file_path)
    G = nx.read_graphml(BytesIO(response.content))
    network_plot = visualize_network(G, centrality_measure, community_detection)
    stats_table = display_statistics(G)
    return pn.Row(
        pn.Column(file_selector, centrality_selector, community_toggle, stats_table),
        pn.Column(network_plot)
    )

app = pn.bind(update_app,
        file_path=file_selector.param.value, 
        centrality_measure=centrality_selector.param.value, 
        community_detection=community_toggle.param.value)



# Display the app
layout = pn.Row(app)
layout.servable()

await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    state.curdoc.apply_json_patch(patch.to_py(), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()