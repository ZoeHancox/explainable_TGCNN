# Graph activation code

import pandas as pd
import tensorflow as tf
from src import utils
import matplotlib.pyplot as plt


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