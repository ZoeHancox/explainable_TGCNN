import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import random
import string
import statistics
from src import create_fake_patients, whole_model_demographics_gradcam, graph_plot, plot_feature_value
import tensorflow as tf
from csv import writer
import matplotlib.pyplot as plt
import math


def generate_filt_sequences(stride:int, filter_size:int, feature_map_size:int) -> list:
    """Return a list of indices that the filter covers in one step. e.g. if the filter is
    size 4 and the stride is one the indices cover timesteps [1,2,3,4] at step one, then 
    [2,3,4,5] at step two...

    Args:
        stride (int): filter stride.
        filter_size (int): size of CNN filter.
        feature_map_size (int): length of the feature map.

    Returns:
        list: list of lists with timestep indices per filter pass.
    """
    sequences = []

    for t in range(feature_map_size):
        start_idx = t * stride + 1
        
        end_idx = start_idx + filter_size
        
        # Generate the list of indices for this timestep
        indices = list(range(start_idx, end_idx))
        
        sequences.append(indices)
    
    return sequences

def find_indices_of_value(list_of_lists, value):
    # find the indices of lists containing the value
    return [index for index, sublist in enumerate(list_of_lists) if value in sublist]


def calc_local_map(model, grads:tf.Tensor, only_pos:bool=True, filt_num:int=None) -> np.array:
    """Calculate the graph-Grad-CAM localisation map by performing global
    average pooling to get the weightings for each filter, then get the weighted 
    activation map before passing this through a ReLU to only focus on features 
    with a positive influence on the class of interest.

    Can choose to look at one filter rather than all.

    Args:
        model (tf.model): trained TG-CNN model.
        grads (tf.Tensor): gradients from the TG-CNN model, the CNN output with
                        respect to the model output.
        only_pos (bool): if True then use ReLU to focus only on features with a positive
                        influence on the class, otherwise get the absolute values.
        filt_num (int): filter number to get the value of, if None then use all.

    Returns:
        np.array: weighted feature map values forming localisation map.
    """
    # Get the average gradient value per timestep

    # Global average pooling to get the weight for each filter

    f_map_branch1 = tf.squeeze(model.f_map_branch1)
    k_size = f_map_branch1.shape[1] # length of feature map
    if filt_num == None:
        grads_2D = tf.squeeze(grads)
        grads_2D = np.array(grads_2D)
        #print(grads_2D.shape) # filters, k_size
    
        alphak_lst = [] # alpha value for each filter
        for filt in range(f_map_branch1.shape[0]):
            filt_grad_sum = tf.reduce_sum(grads_2D, axis=1)[filt]
            alpha_k = (1/k_size)*filt_grad_sum
            alphak_lst.append(float(alpha_k))


        # Get 1D localisation map
        alphak_tf = tf.constant(alphak_lst, dtype=tf.float32)
        alphak_tf = tf.reshape(alphak_tf, (-1, 1))

        weighted_f_maps = alphak_tf*f_map_branch1
        weighted_f_map = tf.reduce_sum(weighted_f_maps, axis=0)


    else:
        filt_grads = grads[0, filt_num]
        filt_grad_sum = tf.reduce_sum(filt_grads)
        alpha_k = (1/k_size)*filt_grad_sum

        # Get 1D localisation map
        alphak_tf = tf.constant(alpha_k, dtype=tf.float32)
        weighted_f_map = alphak_tf*f_map_branch1[filt_num]

    if only_pos:
        # ReLU as we only are interested in the features that have a positive influence of the class of interest
        # Turns negative numbers into 0
        l_map = np.array(tf.nn.relu(weighted_f_map))
    else:
        # Get absolute values instead of ReLU as model is binary and negative weights may also have important insights
        l_map = np.array(tf.abs(weighted_f_map))

    return l_map


def calc_timestep_weights(stride:int, filter_size:int, l_map:np.array,
                          max_timesteps:int):
    """Gets the timestep weighting, showing which timesteps were more important during 
    prediction.

    Args:
        stride (int): filter stride.
        filter_size (int): size of CNN filter.
        l_map (np.array): localisation map - weighted feature activation map.
        max_timesteps (int): maximum possible number of timesteps/visits.

    Returns:
        pd.DataFrame: DataFrame with timestep Grad-CAM weighting.
    """
    sequences = generate_filt_sequences(stride, filter_size, l_map.shape[0])

    f_map_indices = []
    for f in range(1, max_timesteps+1): 
        f_map_indices.append(find_indices_of_value(sequences, f))

    timestep_ave_grad_df = pd.DataFrame({
                            'timestep': list(range(max_timesteps, 0, -1)), 
                            'f_map_indices' : f_map_indices 
                                        })

    #index_values = timestep_ave_grad_df['f_map_indices'].values

    timestep_ave_grads_lst = []
    for index, row in timestep_ave_grad_df.iterrows():
        grad_lst_per_timestep = []
        for i in row['f_map_indices']:
            grad_lst_per_timestep.append(l_map[i])
        # get the mean of the l_map for each time_step
        timestep_ave_grads_lst.append(statistics.mean(grad_lst_per_timestep))

    timestep_ave_grad_df['timestep_ave_grad'] = timestep_ave_grads_lst


    timestep_ave_grad_df['x'] = timestep_ave_grad_df['timestep']

    # nodes in position x = 0 is the same as nodes in position x = 1 so can just duplicate the value here
    # one more column of nodes exists in the graph than edges
    new_row_dup = {'timestep': 0, 'f_map_indices': timestep_ave_grad_df['f_map_indices'].iloc[-1], 'timestep_ave_grad': timestep_ave_grad_df['timestep_ave_grad'].iloc[-1], 'x':0}

    timestep_ave_grad_df.loc[len(timestep_ave_grad_df)] = new_row_dup

    return timestep_ave_grad_df

def map_read_code_labels(pos_df:pd.DataFrame, read_code_map_df:pd.DataFrame, timestep_ave_grad_df:pd.DataFrame):
    """Map the Read Codes and descrptions to the node numbers.

    Args:
        pos_df (pd.DataFrame): DataFrame providing positions for nodes.
        read_code_map_df (pd.DataFrame): DataFrame with the node numbers, Read Codes and descriptions.
        timestep_ave_grad_df (pd.DataFrame): DataFrame with timestep Grad-CAM weighting.

    Returns:
        pd.DataFrame: DataFrame with node positions and labels.
    """
    # Extract the numeric part from the 'node' column in pos_df
    pos_df['node_num'] = pos_df['node'].str.extract('(\d+)').astype(int)

    read_code_pos_df = pos_df.merge(read_code_map_df, on='node_num', how='left')

    # merge the x cols in timestep_ave_grad_df with read_code_pos_df x col to assign grad strength to nodes
    read_code_pos_df = read_code_pos_df.merge(timestep_ave_grad_df, on='x', how='left')

    # # normalise (min-max) the timestep_ave_grad column (needed as not all of the timesteps are used and should give better y axis values)
    # read_code_pos_df['perc_timestep_infl'] = (read_code_pos_df['timestep_ave_grad'] - read_code_pos_df['timestep_ave_grad'].min()) / (read_code_pos_df['timestep_ave_grad'].max() - read_code_pos_df['timestep_ave_grad'].min())
    
    # Return the percentage contribution of each timestep so all timestep values sum to 1
    read_code_pos_df['perc_timestep_infl'] = (read_code_pos_df['timestep_ave_grad'].abs() / read_code_pos_df['timestep_ave_grad'].abs().sum())*100

    read_code_pos_df.fillna(0, inplace=True) # Turn NaNs to 0 for cases where the nodes did not influence prediction
    
    return read_code_pos_df


def create_edge_pos_df(edges_df:pd.DataFrame, pos_df:pd.DataFrame):
    """Merged the edges_df and pos_df to get the coordinates for the edges.

    Args:
        edges_df (pd.DataFrame): DataFrame with columns start_node, end_node and time_between.
        pos_df (pd.DataFrame): DataFrame with columns: node	x, cumulative_count, max_codes_per_visit, y, node_num.

    Returns:
        pd.DataFrame: DataFrame with edge coordinates included.
    """
    # merge edges_df with pos_df on start_node to get x0 and y0
    edge_pos_df = edges_df.merge(pos_df, how='left', left_on='start_node', right_on='node')
    edge_pos_df = edge_pos_df.rename(columns={'x': 'x0', 'y': 'y0'}).drop(columns=['node', 'cumulative_count', 'max_codes_per_visit'])

    # merge the result with pos_df again on end_node to get x1 and y1
    edge_pos_df = edge_pos_df.merge(pos_df, how='left', left_on='end_node', right_on='node')
    edge_pos_df = edge_pos_df.rename(columns={'x': 'x1', 'y': 'y1'}).drop(columns=['node', 'cumulative_count', 'max_codes_per_visit'])
    return edge_pos_df


def text_color_mapping(numbers):
    # Map the text colour based on the colour of the node so that it can be read
    return ['black' if num < 50 else 'white' for num in numbers]

def plot_gradcam_plotly(edge_pos_df:pd.DataFrame, pos_df:pd.DataFrame, read_code_pos_df:pd.DataFrame,
                        years_in_advance:str, logits:tf.Tensor, outcome:str, filename:str):
    """_summary_

    Args:
        edge_pos_df (pd.DataFrame): _description_
        pos_df (pd.DataFrame): _description_
        read_code_pos_df (pd.DataFrame): _description_
        years_in_advance (str): _description_
        logits (tf.Tensor): _description_
        outcome (str): _description_
        filename (str): Name to save the file as, this is suffixed with '_plot.html'.
    """
    edge_x = []
    edge_y = []

    for _, row in edge_pos_df.iterrows():
        edge_x.extend([row['x0'], row['x1'], None])
        edge_y.extend([row['y0'], row['y1'], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',  # Disable hover info for lines
        mode='lines'
    )

    node_x = pos_df['x'].tolist()
    node_y = pos_df['y'].tolist()
    node_labels = read_code_pos_df['ReadCode'].tolist()

    read_code_desc_list = read_code_pos_df['ReadCode_descript'].tolist()
    perc_infl = read_code_pos_df['perc_timestep_infl'].tolist()
    # Add the percentage to each hover
    node_hover_text = [f"{read_code_desc_list[i]} \nInfluence visit has on prediction: {round(perc_infl[i], 2)}%" for i in range(len(read_code_desc_list))]
    #node_hover_text = read_code_pos_df['ReadCode_descript'].tolist()

    text_colors = text_color_mapping(read_code_pos_df['perc_timestep_infl'].to_list())

    # Create node_trace with static labels and hover text
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,  # Static labels for nodes
        textposition='middle center', 
        textfont=dict(
            size=10,
            color=text_colors  # Set color for text
        ),
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color='black',
            size=45,
            colorbar=dict(
                thickness=15,
                title='Influence Visit Had on Risk Prediction (%)',
                xanchor='left',
                titleside='right'
            ),
            line_width=2,
            cmin=0, # min of colorbar to be 0
            cmax=100 # max of colorbar to be 100
        ),
        hovertext=node_hover_text
    )

    node_colors = read_code_pos_df['perc_timestep_infl'].to_list()

    # This shouldn't happen if we don't normalise
    # if True in np.isnan(np.array(node_colors)):
    #     node_colors = [1]*len(node_colors)
        
    node_trace.marker.color = node_colors


    # Calculate midpoints for edge label annotations
    annotations = []
    for _, row in edge_pos_df.iterrows():
        mid_x = (row['x0'] + row['x1']) / 2
        mid_y = (row['y0'] + row['y1']) / 2
        annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=f"{int(round(row['time_between']*30.4167,0))} days",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
        )

    if outcome == 'hip':
        out_print = f'Patient will likely need a hip replacement in {years_in_advance} years time.'
    else:
        out_print = f'It is unlikely this patient will need a hip replacement in {years_in_advance} years time.'
    stream_num = '1'

    proba_of_replace = tf.nn.sigmoid(logits)
    #proba_of_replace = 1/ (1+np.exp(logits)) # use sigmoid to convert logits to probs
    
    # if ((proba_of_replace.item() > 0.5) and (outcome != 'hip')) or ((proba_of_replace.item()) < 0.5) and (outcome == 'hip'):
    #     print("Model predicted incorrectly")
    if outcome == 'hip':
        true_out = f'A hip replacement was needed.'
    else:
        true_out = f'A hip replacement was not needed.'
        
    # if proba_of_replace.item() > 0.5:
    #     model_pred = 'will need a hip replacement'
    # else:
    #     model_pred = 'will not need a hip replacement'
    # out_print = f"The model predicts that this person {model_pred} in {years_in_advance} years time."
    # print(logits)
    # print(tf.squeeze(proba_of_replace).numpy())    
    annotations.append(dict(
                        text=(
                            f"The models predicts the probability of this patient needing a replacement is {round(tf.squeeze(proba_of_replace).numpy()*100,2)}%.<br>"
                            f"The patient's true outcome: {true_out}"),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.002))
    annotations.append(dict(
                        text=("Most Recent"),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.95, y=0.95))
    annotations.append(dict(
                        text=("Most Distant"),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.05, y=0.95))
    annotations.append(dict(
                        text=("------------------------------------>"),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.95))    
    

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Patient Pathway Graph and Influence on Model Prediction.",
                        titlefont_size=14,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=15, l=10, r=5, t=40),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    

    pio.write_html(fig, file=filename+"_plot.html", auto_open=True)
    fig.show()