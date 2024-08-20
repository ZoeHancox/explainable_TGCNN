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
    """Get the maximum, median or mean of each feature map for each patient alongside the patients true outcome.

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
            f_map_shape_1 = feature_maps.shape[1]
            f_map_shape_2 = feature_maps.shape[2]
            f_map_index_rows = tf.reshape(tf.transpose(feature_maps), [1, f_map_shape_2, f_map_shape_1])
            x_sorted = tf.sort(f_map_index_rows, axis=2)
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

def act_diff(replacement_true_lst:list, max_w_filt_lst:list, filt_nums:list, show_plot:bool=True):
    """Calculate the difference between the two Hip Replacement classes for each Filter

    Args:
        replacement_true_lst (list): binary whether someone has a replacement or not.
        max_w_filt_lst (list): maximum value in feature map.
        filt_nums (list): number of filters.
        show_plot (bool): If True print the plot.

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
    plt.savefig("feature_map_plots/filter_difference.png", bbox_inches='tight')
    if show_plot:
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

def choose_feat_map(model, fm_type:str, mean_activation_df:pd.DataFrame, feat_map_num:int=None) -> np.array:
    """Get mean or median of feature maps combined or get the feature map with the highest 
    difference in activation between the two classes. Alternatively just return one feature map.

    Args:
        model (tf.Model): trained TG-CNN model.
        fm_type (str): which operation to perform to get the activation map to use for the weights.
        mean_activation_df (pd.DataFrame): dataframe with the difference in activation for each filter and class.
        feat_map_num (int): filter to return.

    Returns:
        np.array: activation map to use for the graph visual weights.
    """
    
    #  get the feature maps
    f_maps = tf.squeeze(model.f_map_branch1)
    # get the average or median across the feature maps
    if fm_type == 'mean':
        comb_f_maps = tf.reduce_sum(f_maps, axis=0)
    elif fm_type == 'median': # if the violin plots are not normally distributed then take the median
        f_maps = model.f_map_branch1
        f_map_shape_1 = model.f_map_branch1.shape[1]
        f_map_shape_2 = model.f_map_branch1.shape[2]
        f_map_index_rows = tf.reshape(tf.transpose(f_maps), [1, f_map_shape_2, f_map_shape_1])
        x_sorted = tf.sort(f_map_index_rows, axis=2)
        n = x_sorted.shape[2]

        # Calculate the median index
        middle = n // 2

        # Compute the median along the specified axis
        # If the size of the axis is odd, take the middle element
        # If even, average the two middle elements
        comb_f_maps = tf.cond(
            tf.equal(n % 2, 1),
            lambda: tf.gather(x_sorted, middle, axis=2),
            lambda: tf.reduce_mean(tf.gather(x_sorted, [middle - 1, middle], axis=2), axis=2)
        )
        comb_f_maps = tf.squeeze(comb_f_maps)
    elif fm_type == 'largest':
        filt_num = find_max_act_filt(mean_activation_df)
        comb_f_maps = f_maps[filt_num-1] # minus 1 as we don't have a filter called 0
    elif fm_type == 'single':
        comb_f_maps = f_maps[feat_map_num-1]
    else:
        raise ValueError(f"Unsupported metric: '{fm_type}'. Expected one of: 'mean', 'largest', 'single', or 'median'.")

    return np.array(comb_f_maps)


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

def get_and_reshape_filt(filters_4d:np.array, max_act_filt_num:int, filt_type:str, filter_num:int=None) -> tf.Tensor:
    """Take the filters and select the filter with the highest activation and reshape to match 
    the patient graph direction i.e. most recent events on the right.
    Use for edge activation graph.

    Args:
        filters_4d (np.array): 4D stack of 3D filters.
        max_act_filt_num (int): filter number with the highest activation difference between the two classes.
        filt_type (str): which operation to perform across the filters. Can be 'max', 'median' or 'mean'.
        filter_num (int): used if the filt_type is 'single'.

    Returns:
        tf.Tensor: single filter with the same ordering as the patient graph tensor.
    """
    if filt_type == 'max':
        # get the filter with the largest activation difference between classes
        f = filters_4d[max_act_filt_num-1] # minus 1 as we don't have a filter called 0
    elif filt_type == 'median':
        f = np.median(filters_4d, axis=0)
    elif filt_type == 'mean':
        f = np.mean(filters_4d, axis=0)
    elif filt_type == 'single':
        f = filters_4d[filter_num-1]

    f = np.flip(f, axis=0) # flip the filter so the most recent event is at the end rather than the start
    f = tf.transpose(f, perm=[2, 1, 0]) # reorder filter
    return f



def filt_times_pat(f:tf.Tensor, dense_tensor:tf.Tensor, filter_size:int, max_timesteps:int, stride:int) -> tf.Tensor:
    """Get the filter element-wise multiplied by the patient graph, with a sliding window taking the mean 
    of each pass.
    This is the element-wise multiplication between the slices of the filter and the patient graph
    with the mean for each group of operations.
    The values in this tensor should be used for the edge weightings.

    Args:
        f (tf.Tensor): filter from 3D CNN.
        dense_tensor (tf.Tensor): patient 3D tensor representing graph.
        filter_size (int): number of timesteps in the 3D CNN filter.
        max_timesteps (int): number of timesteps covered by the patient graph.
        stride (int): return error if the stride is not 1.

    Returns:
        tf.Tensor: Edge activated patient graph.
    """
    if stride != 1:
        raise ValueError(f"This calculation requires the filter stride to be 1, you've supplied a stride of {stride}.")
    # First sort the middle of the pat graph which has all the filter timesteps passing over
    tensors = []
    for t in range(filter_size):
        graph_ends = max_timesteps-2*(filter_size-1)
        f_repeat = tf.repeat(tf.expand_dims(f[t,:,:], axis=0), repeats=graph_ends, axis=0) # times by two for both ends
        mid_graph = f_repeat * dense_tensor[filter_size-1:max_timesteps-filter_size+1, :, :]
        tensors.append(mid_graph)

    stacked_tensors = tf.stack(tensors, axis=0)
    mean_tensor_mid = tf.reduce_mean(stacked_tensors, axis=0)  


    # sort out the parts of the graph where they don't have the full filter pass over them
    # BEGINNING FIRST
    num_to_append = filter_size - 1

    # Generate slices_indices dynamically
    slices_indices = []
    for i in range(num_to_append):
        indices = list(range(min(f.shape[0], num_to_append - i)))
        indices.append(filter_size - 2 - i)  # Add the corresponding index for patient graph
        slices_indices.append(indices)

    # calculate the means
    for indices in slices_indices:
        # Perform element-wise multiplication and compute the mean
        multiplied_slices = [f[i, :, :] * dense_tensor[indices[-1], :, :] for i in indices[:-1]]
        mean_value = tf.reduce_mean(multiplied_slices, axis=0)
        mean_tensor_mid = tf.concat([tf.expand_dims(mean_value, axis=0), mean_tensor_mid], axis=0)

    # END NEXT
    base_list = list(range(1, filter_size))
    negative_numbers = [-i for i in range(len(base_list), 0, -1)]

    slices_indices = []
    for i in range(len(base_list)):
        sublist = base_list[i:] + [negative_numbers[i]]
        slices_indices.append(sublist)

    # calculate the means
    for indices in slices_indices:
        # Perform element-wise multiplication and compute the mean
        multiplied_slices = [f[i, :, :] * dense_tensor[indices[-1], :, :] for i in indices[:-1]]
        mean_value = tf.reduce_mean(multiplied_slices, axis=0)
        # Add the tensor to the end of the mid tensor
        mean_tensor_mid = tf.concat([mean_tensor_mid, tf.expand_dims(mean_value, axis=0)], axis=0)

    return mean_tensor_mid

def create_edges_df_ga(patient_graph:np.array, edge_w_graph:tf.Tensor) -> pd.DataFrame:
    """Create a DataFrame of the edges of the patient graph, including the start and end nodes, the time 
    between visits, and the edge weight. For edge activation model.

    Args:
        patient_graph (np.array): 3D numpy array showing the patients health codes over time.
        edge_w_graph (tf.Tensor): 

    Returns:
        DataFrame: with columns start_node, end_node, time_between.
    """

    # Get the indices of non-zero elements
    patient_graph = patient_graph.numpy()
    edge_w_graph = np.array(edge_w_graph)
    t_indices, i_indices, j_indices = np.nonzero(patient_graph)
    non_zero_values = []
    edge_weights = []
    for t, i, j in zip(t_indices, i_indices, j_indices):
        non_zero_values.append(patient_graph[t, i, j])
        edge_weights.append(np.maximum(0, edge_w_graph[t, i, j])) # maximum compares 0 to value and gives a ReLU return
    if not non_zero_values:
        print("Error: Graph does not contain any values.")

    # Calculate start_node_v and end_node_v
    start_node_v = np.where(t_indices == 0, 0, t_indices)
    end_node_v = start_node_v + 1

    edges_df = pd.DataFrame({
        'start_node': [f'{i}_v{start_v}' for i, start_v in zip(i_indices, start_node_v)],
        'end_node': [f'{j}_v{end_v}' for j, end_v in zip(j_indices, end_node_v)],
        'time_between': non_zero_values,
        'edge_weights': edge_weights
    })
    
    return edges_df

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

    edge_pos_df['edge_weight_perc'] = (edge_pos_df['edge_weights'].abs()/edge_pos_df['edge_weights'].abs().sum())*100
    edge_pos_df.fillna(0, inplace=True) # Turn NaNs to 0 for cases where the nodes did not influence prediction

    return edge_pos_df


def plot_edge_act_plotly(edge_pos_df:pd.DataFrame, pos_df:pd.DataFrame, read_code_pos_df:pd.DataFrame,
                        years_in_advance:str, logits:tf.Tensor, outcome:str, filename:str, html_open:bool):
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
        html_open (bool): If True open HTML plotly graph in new tab.
    """
    cmap = plt.cm.viridis_r
    edge_influ_perc = (edge_pos_df['edge_weight_perc'] / 100).tolist()
    # Map each percentage to a color in the reversed Viridis colormap (larger numbers = darker)
    edge_colors = [cmap(p) for p in edge_influ_perc]
    edge_hex_colors = [mcolors.to_hex(color) for color in edge_colors]

    cmap = plt.cm.Greys_r
    edge_text_colors = [cmap(p) for p in edge_influ_perc]
    edge_text_hex_colors = [mcolors.to_hex(color) for color in edge_text_colors]

    edge_traces = []
    midpoint_traces = []  # To store the midpoint traces
    for i, row in edge_pos_df.iterrows():
        # Edge trace
        edge_trace = go.Scatter(
            x=[row['x0'], row['x1'], None],
            y=[row['y0'], row['y1'], None],
            line=dict(
                width=8,
                color=edge_hex_colors[i],  # Map colors directly
            ),
            hoverinfo='none',  # Disable hover info for lines
            mode='lines',
        )
        edge_traces.append(edge_trace)

        # Midpoint trace (invisible node)
        mid_x = (row['x0'] + row['x1']) / 2
        mid_y = (row['y0'] + row['y1']) / 2
        midpoint_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(
                size=0,  # Invisible node
                color=edge_hex_colors[i]  # Same color as edge
            ),
            textfont=dict(
            size=1,
            color='black'  # Set color for text depending on background color
            ),
            hoverinfo='text',  # Enable hover info for midpoint node
            hovertext=f"Influence Read Code pair has on prediction: {round(row['edge_weight_perc'], 2)}%"
        )
        midpoint_traces.append(midpoint_trace)

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
                title='Influence Read Codes Had on Risk Prediction (%)',
                xanchor='left',
                titleside='right'
            ),
            line_width=1,
            cmin=0,  # min of colorbar to be 0
            cmax=100  # max of colorbar to be 100
        ),
        hovertext=node_hover_text
    )

    

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

    proba_of_replace = tf.nn.sigmoid(logits)

    # Add final annotation with the model prediction
    annotations.append(dict(
        text=(
            f"The model predicts the probability of this patient needing a replacement is {round(tf.squeeze(proba_of_replace).numpy() * 100, 2)}%.<br>"
            f"The patient's true outcome: {true_out}"
        ),
        showarrow=False,
        xref="paper", yref="paper",
        x=0.5, y=-0.002
    ))

    annotations.append(dict(
        text=("Most Recent"),
        showarrow=False,
        xref="paper", yref="paper",
        x=0.95, y=0.95
    ))
    annotations.append(dict(
        text=("Most Distant"),
        showarrow=False,
        xref="paper", yref="paper",
        x=0.05, y=0.95
    ))

    annotations.append(dict(
        text=("------------------------------------>"),
        showarrow=False,
        xref="paper", yref="paper",
        x=0.5, y=0.95
    ))

    # Combine edge traces, midpoint traces, and node trace
    fig = go.Figure(data=edge_traces + midpoint_traces + [node_trace],
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

    # Save the figure to an HTML file and display it
    pio.write_html(fig, file=filename + "_plot.html", auto_open=html_open)
    fig.show()
