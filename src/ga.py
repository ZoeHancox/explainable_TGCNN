# Graph activation code

import pandas as pd
import tensorflow as tf
from src import utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np


def get_act_metric_per_feat(model, num_filters:int, num_patients:int, pat_df:pd.DataFrame, 
                         max_event_codes:int, hip_or_knee:str, metric:str):
    """Get the maximum of each feature map for each patient alongside the patients true outcome.

    Args:
        model (tf.Model): trained TG-CNN model.
        num_filters (int): number of filters used in the CNN.
        num_patients (int): number of patients.
        pat_df (pd.DataFrame): DataFrame of all the patients graph constructors and demographics.
        max_event_codes (int): maximum number of Read Codes/nodes.
        hip_or_knee (str): 'hip' or 'knee' depending on the model.
        metric (str): can be 'max', 'mean' or 'median'.

    Returns:
        list, list, list: list of classes per patient, list of max activation 
        per feature map, list of the filter numbers.
    Raises:
        ValueError: If the metric is not one of the expected values.
    """
    replacement_true_lst = []
    max_w_filt_lst = []
    filt_nums = list(range(1, num_filters+1))  # Since filt_nums is always 1 to num_filters for each patient

    for i in range(num_patients):
        input_3d, input_4d, demo_tensor, outcome, outcome_bin = utils.return_pat_from_df(pat_df, max_event_codes, hip_or_knee, i)
        
        dense_tensor = tf.sparse.to_dense(input_3d)
        dense_tensor = tf.transpose(dense_tensor, perm=[2, 1, 0])
        dense_tensor = tf.reverse(dense_tensor, axis=[0])
        
        logits = model(input_4d, demo_tensor, training=False)
        
        feature_maps = model.f_map_branch1
        
        if metric == 'max':
            # Get the maximum value from each filter and apply ReLU
            w_per_filt = tf.reduce_max(feature_maps, axis=2)
        elif metric == 'mean':
            w_per_filt = tf.reduce_mean(feature_maps, axis=2)
        elif metric == 'median':
            x_sorted = tf.sort(feature_maps, axis=2)
            n = x_sorted.shape[2]

            # Calculate the median index (integer division)
            middle = n // 2

            # Compute the median along the specified axis
            # If the size of the axis is odd, take the middle element
            # If even, average the two middle elements
            w_per_filt = tf.cond(
                tf.equal(n % 2, 1),
                lambda: tf.gather(x_sorted, middle, axis=2),
                lambda: tf.reduce_mean(tf.gather(x_sorted, [middle - 1, middle], axis=2), axis=2)
            )
        else:
            raise ValueError(f"Unsupported metric: '{metric}'. Expected one of: 'mean', 'max', or 'median'.")



        w_per_filt_relu = tf.nn.relu(w_per_filt)
        
        # Convert to list and extend results directly
        w_per_filt_relu_np = w_per_filt_relu.numpy().flatten().tolist()
        
        replacement_true_lst.extend([outcome_bin] * num_filters)
        max_w_filt_lst.extend(w_per_filt_relu_np)

    # filt_nums list should be extended `num_patients` times.
    filt_nums = filt_nums * num_patients

    return replacement_true_lst, max_w_filt_lst, filt_nums

def act_diff(replacement_true_lst:list, max_w_filt_lst:list, filt_nums:list):
    """Calculate the difference between the two Hip Replacement classes for each Filter

    Args:
        replacement_true_lst (list): binary whether someone has a replacement or not.
        max_w_filt_lst (list): maximum value in feature map.
        filt_nums (list): number of filters.

    Returns:
        pd.DataFrame: DataFrame with filters and the respective difference in the max activation 
        values between the two classes.
    """
    data = {
        'Hip Replacement': replacement_true_lst,
        'Max Activation': max_w_filt_lst,
        'Filter': filt_nums
    }

    filt_act_df = pd.DataFrame(data)

    mean_activation = filt_act_df.groupby(['Filter', 'Hip Replacement'])['Max Activation'].mean()

    mean_activation_df = mean_activation.to_frame()
    mean_activation_df.reset_index(inplace=True)

    # Calculate the difference between the two Hip Replacement rows for each Filter
    mean_activation_df['Difference'] = mean_activation_df.groupby('Filter')['Max Activation'].diff()
    mean_activation_df['Difference'] = mean_activation_df['Difference'].abs()

    mean_activation_df = mean_activation_df[['Filter', 'Difference']].dropna().reset_index(drop=True)

    mean_activation_df.sort_values(by='Difference', ascending=False)

    mean_activation_df['Filter'] = mean_activation_df['Filter'].astype(int)

    plt.figure(figsize=(10, 6))
    #plt.plot(mean_activation_df['Filter'], mean_activation_df['Difference'], marker='.', linestyle='-')
    plt.bar(mean_activation_df['Filter'], mean_activation_df['Difference'], color='turquoise')
    plt.xlabel('Filter')
    plt.ylabel('Mean Difference Between Class 0 and Class 1')
    plt.title('Difference in Activation per Class for Each Filter')
    plt.xticks(mean_activation_df['Filter'], rotation=45)  # Set x-axis ticks to integer values with rotation
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

    return mean_activation_df

def find_max_act_filt(mean_activation_df:pd.DataFrame) -> int:
    """Find the filter which has the largest activation difference between the two classes.

    Args:
        mean_activation_df (pd.DataFrame): df containing the difference in activation between classes 
                                        for each filter.

    Returns:
        int: filter number with the largest activation difference.
    """
    max_idx = mean_activation_df['Difference'].idxmax()
    filt_num = mean_activation_df.loc[max_idx, 'Filter']
    return filt_num

def make_filts_4d(filters:tf.Tensor, filter_size:int, max_event_codes:int) -> np.array:
    """Take the flattened filters from the model and convert them to 4D for matrix multiplication 
    with patient graphs. 

    Args:
        filters (tf.Tensor): 2D flattened filters.
        filter_size (int): size of the 3D CNN filters.
        max_event_codes (int): number of nodes/event codes in the model.

    Returns:
        np.array: 4D 3D CNN filters, each filter is 3D and stacked together into 4D.
    """
    filters_trans = tf.transpose(filters, [1,0]) # change filters to have one 2D array per filter rather than splitting the array 

    filters_4d = []
    for f in filters_trans:   
        filters_3d=[]
        #loop through one z/timestep at a time
        for i in range(filter_size):
            # Select numbers for the current tensor
            selected_numbers = f[i::filter_size]
            # Convert the selected numbers into a NumPy array and reshape it to a tensor
            oned_tensor = np.array(selected_numbers)#.reshape(-1, 1)
            tensor = oned_tensor.reshape(max_event_codes, max_event_codes)
            filters_3d.append(np.array(tensor))
            
        filters_3d = np.array(filters_3d)
        filters_3d = filters_3d.reshape(max_event_codes,max_event_codes,filter_size)
        filters_4d.append(filters_3d)

    filters_4d = np.array(filters_4d) 
    return filters_4d

def get_and_reshape_filt(filters_4d:np.array, max_act_filt_num:int) -> tf.Tensor:
    """Take the filters and select the filter with the highest activation and reshape to match 
    the patient graph direction i.e. most recent events on the right.

    Args:
        filters_4d (np.array): 4D stack of 3D filters.
        max_act_filt_num (int): filter number with the highest activation difference between the two classes.

    Returns:
        tf.Tensor: single filter with the same ordering as the patient graph tensor.
    """
    # get the filter with the largest activation difference between classes
    max_act_filt = filters_4d[max_act_filt_num-1] # minus 1 as we don't have a filter called 0
    f = np.flip(max_act_filt, axis=0) # flip the filter so the most recent event is at the end rather than the start
    f = tf.transpose(f, perm=[2, 1, 0]) # reorder filter
    return f


def map_read_code_labels(pos_df:pd.DataFrame, read_code_map_df:pd.DataFrame) -> pd.DataFrame:
    """Map the Read Codes and descrptions to the node numbers.

    Args:
        pos_df (pd.DataFrame): DataFrame providing positions for nodes.
        read_code_map_df (pd.DataFrame): DataFrame with the node numbers, Read Codes and descriptions.

    Returns:
        pd.DataFrame: DataFrame with node positions and labels.
    """
    # Extract the numeric part from the 'node' column in pos_df
    pos_df['node_num'] = pos_df['node'].str.extract('(\d+)').astype(int)

    read_code_pos_df = pos_df.merge(read_code_map_df, on='node_num', how='left')

    # # Return the percentage contribution of each timestep so all timestep values sum to 1
    # read_code_pos_df['perc_timestep_infl'] = (read_code_pos_df['timestep_ave_grad'] / read_code_pos_df['timestep_ave_grad'].sum())*100
    
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

    edge_pos_df['edge_weight_perc'] = (edge_pos_df['edge_weights']/edge_pos_df['edge_weights'].sum())*100
    return edge_pos_df


def plot_edge_act_plotly(edge_pos_df:pd.DataFrame, pos_df:pd.DataFrame, read_code_pos_df:pd.DataFrame,
                        years_in_advance:str, logits:tf.Tensor, outcome:str, filename:str):
    """Create plotly graph that colors the edges depending on how much influence the connecting Read Codes 
    have on model prediction.

    Args:
        edge_pos_df (pd.DataFrame): _description_
        pos_df (pd.DataFrame): _description_
        read_code_pos_df (pd.DataFrame): _description_
        years_in_advance (str): _description_
        logits (tf.Tensor): _description_
        outcome (str): _description_
        filename (str): Name to save the file as, this is suffixed with '_plot.html'.
    """
    cmap = plt.cm.viridis_r
    edge_influ_perc = (edge_pos_df['edge_weight_perc']/100).tolist()
    # Map each percentage to a color in the reversed Viridis colormap (larger numbers = darker)
    edge_colors = [cmap(p) for p in edge_influ_perc]
    
    # Convert RGBA colors to hex format
    edge_hex_colors = [mcolors.to_hex(color) for color in edge_colors]
    
    edge_traces = []
    for i, row in edge_pos_df.iterrows():
        edge_trace = go.Scatter(
        x=[row['x0'], row['x1'], None],
        y=[row['y0'], row['y1'], None],
        line=dict(
            width=8,
            color=edge_hex_colors[i],  # Map colors directly
            #colorscale='Viridis',  # Use a predefined colorscale
        ),
        hoverinfo='none',  # Disable hover info for lines
        mode='lines'
        )
        edge_traces.append(edge_trace)

    node_x = pos_df['x'].tolist()
    node_y = pos_df['y'].tolist()
    node_labels = read_code_pos_df['ReadCode'].tolist()
    node_hover_text = read_code_pos_df['ReadCode_descript'].tolist()

    # Create node trace with a single, uniform color
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,  # Static labels for nodes
        textposition='middle center',
        textfont=dict(
            size=10,
            color='black'  # Set color for text
        ),
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color='white',
            size=45,
            colorbar=dict(
                thickness=15,
                title='Influence Visit Had on Risk Prediction (%)',
                xanchor='left',
                titleside='right'
            ),
            line_width=1,
            cmin=0, # min of colorbar to be 0
            cmax=100 # max of colorbar to be 100
        ),
        hovertext=node_hover_text
    )


    cmap = plt.cm.Greys_r
    edge_influ_perc = (edge_pos_df['edge_weight_perc']/100).tolist()
    edge_text_colors = [cmap(p) for p in edge_influ_perc]
    edge_text_hex_colors = [mcolors.to_hex(color) for color in edge_text_colors]

    # Calculate midpoints for edge label annotations
    annotations = []
    for i, row in edge_pos_df.iterrows():
        mid_x = (row['x0'] + row['x1']) / 2
        mid_y = (row['y0'] + row['y1']) / 2
        annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=f"{int(round(row['time_between'] * 30.4167, 0))} days",
                showarrow=False,
                font=dict(size=10, color=edge_text_hex_colors[i]),
                align="center"
            )
        )

    # Determine the outcome text
    if outcome == 'hip':
        true_out = 'A hip replacement was needed.'
    else:
        true_out = 'A hip replacement was not needed.'

    proba_of_replace = 1 / (1 + np.exp(logits))  # use sigmoid to convert logits to probs

    # Add final annotation with the model prediction
    annotations.append(dict(
        text=f"The model predicts the probability of this patient needing a replacement is {round(proba_of_replace.item() * 100, 2)}%. \nThe patient's true outcome: {true_out}",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.005, y=-0.002
    ))


    edge_traces.append(node_trace)
    # Create the figure
    fig = go.Figure(data=edge_traces,                
                    layout=go.Layout(
                        title=f'Graph Visualisation of Patients Pathway and Connections Associated to Hip Replacement - Stream 1.',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=15, l=10, r=5, t=50),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Save the figure to an HTML file and display it
    pio.write_html(fig, file=filename + "_plot.html", auto_open=True)
    fig.show()
