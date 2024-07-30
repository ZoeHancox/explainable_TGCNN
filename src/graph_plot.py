# Code for all of the graph visualisation types

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import webbrowser
import os
import pandas as pd
import random
import string
import statistics
from src import create_fake_patients, whole_model_demographics_gradcam
from tensorflow import keras
import tensorflow as tf
from csv import writer
import seaborn as sns
import matplotlib.pyplot as plt
print("tensorflow version:", tf. __version__)
tf.config.list_physical_devices()

def create_edges_df_gc(patient_graph:np.array):
    """Create a DataFrame of the edges of the patient graph, including the start and end nodes and the time 
    between visits. For Grad-CAM model.

    Args:
        patient_graph (np.array): 3D numpy array showing the patients health codes over time.

    Returns:
        DataFrame: with columns start_node, end_node, time_between.
    """

    # Get the indices of non-zero elements
    patient_graph = np.round(patient_graph.numpy(), 4)
    t_indices, i_indices, j_indices = np.nonzero(patient_graph)
    non_zero_values = []
    for t, i, j in zip(t_indices, i_indices, j_indices):
        non_zero_values.append(patient_graph[t, i, j])
    #print("Non-zero values", non_zero_values)
    if not non_zero_values:
        print("Error: Subgraph does not contain any values.")

    # Calculate start_node_v and end_node_v
    start_node_v = np.where(t_indices == 0, 0, t_indices)
    end_node_v = start_node_v + 1

    edges_df = pd.DataFrame({
        'start_node': [f'{i}_v{start_v}' for i, start_v in zip(i_indices, start_node_v)],
        'end_node': [f'{j}_v{end_v}' for j, end_v in zip(j_indices, end_node_v)],
        'time_between': non_zero_values
    })
    
    return edges_df

def extract_visit_number(s:str):
    """Extract the visit number from a string

    Args:
        s (str): string to split

    Returns:
        int: visit number
    """
    return int(s.split('_')[1][1:])

def create_position_df_gc(edges_df:np.array):
    """Create a DataFrame to find the number of nodes per visit. For Grad-CAM model.

    Args:
        edges_df (pd.DataFrame): DataFrame with columns start_node, end_node, time_between.

    Returns:
        pd.DataFrame: with columns showing node name (node), visit number (x), node number in visits (cumulative count),
        and maximum codes per visit.
    """
    pos_df = edges_df[['start_node', 'end_node']].stack().drop_duplicates().reset_index(drop=True)
    pos_df = pos_df.to_frame(name='node')
    pos_df['x'] = pos_df['node'].apply(extract_visit_number)
    pos_df['cumulative_count'] = pos_df.groupby('x').cumcount()
    pos_df['max_codes_per_visit'] = pos_df.groupby('x')['cumulative_count'].transform('max') + 1

    return pos_df

def generate_pos_sequence(x:int):
    """Generate a list of lists to get y coordinate positions for the nodes
     based on the number of events recorded per visit.

    Args:
        x (int): maximum number of nodes in any one visit

    Returns:
        List: List of lists of y coordinates index mapping to the max nodes for
               each visit.
    """
    sequence = []
    for i in range(x):
        if i % 2 == 0:  # Even index, include zero
            sequence.append(list(range(-i // 2, i // 2 + 1)))
        else:  # Odd index, exclude zero
            sublist = list(range(-(i // 2 + 1), i // 2 + 2))
            sublist.remove(0)
            sequence.append(sublist)
    return sequence


def get_pos_y_value_per_node(row, pos_list:list):
    """Get the y position for each node. Use max_codes_per_visit column to select the sublist 
    and the cumulative_count to get the position from the sublist.

    Args:
        row (pd.Series): row of the DataFrame

    Returns:
        int: y coordinate position 
    """
    cum_count = row['cumulative_count']
    max_codes = row['max_codes_per_visit']
    return pos_list[max_codes - 1][cum_count]

def map_y_coord_to_node(pos_df:pd.DataFrame, pos_list:list):
    """Map the y coordinates to the relevant node and correct row.

    Args:
        pos_df (pd.DataFrame): columns with node name and x coordinate.
        pos_list (list of lists): List of lists of y coordinates index mapping to the max nodes for
               each visit.

    Returns:
        pd.DataFrame: dataframe with x and y coordinates for node plotting.
    """
    pos_df['y'] = pos_df.apply(lambda row: get_pos_y_value_per_node(row, pos_list), axis=1)
    return pos_df

def create_pos_dict(pos_df:pd.DataFrame):
    """Make a dictionary with the node name as the key and the x and y coordinates 
    as a tuple value.

    Args:
        pos_df (pd.DataFrame): dataframe with columns for the node name, x coordinates,
        and y coordinates.

    Returns:
        dict: dictionary of node: (x,y)
    """
    # the visit number is x and the y value is the number of nodes with the same visit number
    pos = pos_df.set_index('node')[['x', 'y']].apply(tuple, axis=1).to_dict()
    return pos

def draw_gc_pat_graph(edges_df:pd.DataFrame, pos_dict:pd.DataFrame, graph_name:str, pat_outcome, 
                      stream_num:int=1, years_in_advance:str='5'):
    """Draw an individual patient's subgraph graph using NetworkX.

    Args:
        edges_df (pd.DataFrame): Dataframe with information about edges including: start and end nodes, and edge_label.
        pos_dict (dictionary): dictionary with x and y coordinates for each node:(x,y)
    """
    # Convert df to list of tuples for Networkx
    edges = []
    for _, row in edges_df.iterrows():
        edge = (row['start_node'], row['end_node'], {'edge_label': np.round(row['time_between']*30.4167).astype(int)}) # to get days instead of months * 30.4167
        edges.append(edge)

    G = nx.DiGraph()
    G.add_edges_from(edges)


    edge_labels = {(u, v): G[u][v]['edge_label'] for u, v in G.edges()}

    # Draw the graph
    plt.figure(figsize=(edges_df['start_node'].nunique()+2, 8))
    nx.draw(G, pos_dict, with_labels=True, node_size=3000, node_color="lightblue", edge_color='grey', width=0.5, font_size=10, font_weight="bold", arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=edge_labels, font_color='blue', font_size=10, font_weight='bold')
    if pat_outcome == 'hip':
        out_print = f'Patient will likely need a hip replacement in {years_in_advance} years time'
    else:
        out_print = f'It is unlikely this patient will need a hip replacement in {years_in_advance} years time'
    plt.title(f"Graph Visualisation of Patients Pathway and Connections Associated to Hip Replacement - Stream {stream_num}\n{out_print}")
    plt.savefig(f"documentation/{graph_name}.png", bbox_inches='tight')
    plt.show()
    plt.close()

def create_scrollable_html(image_path, html_path):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            .scrollable-container {{
                width: 100%;
                overflow-x: scroll;
                white-space: nowrap;
            }}
            .scrollable-container img {{
                display: block;
            }}
        </style>
        <title>Graph for Patient Subgraph</title>
    </head>
    <body>
        <div class="scrollable-container">
            <img src="{image_path}" alt="NetworkX Graph">
        </div>
    </body>
    </html>
    """

    with open(html_path, 'w') as file:
        file.write(html_content)