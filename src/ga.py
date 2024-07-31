# Graph activation code

import pandas as pd
import tensorflow as tf
from src import utils


def get_max_act_per_feat(model, num_filters:int, num_patients:int, pat_df:pd.DataFrame, 
                         max_event_codes:int, hip_or_knee:str):
    """Get the maximum o feach feature map for each patient alongside the patients true outcome.

    Args:
        model (tf.Model): trained TG-CNN model.
        num_filters (int): number of filters used in the CNN.
        num_patients (int): number of patients.
        pat_df (pd.DataFrame): DataFrame of all the patients graph constructors and demographics.
        max_event_codes (int): maximum number of Read Codes/nodes.
        hip_or_knee (str): 'hip' or 'knee' depending on the model.

    Returns:
        list, list, list: list of classes per patient, list of max activation 
        per feature map, list of the filter numbers.
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
        
        # Get the maximum value from each filter and apply ReLU
        max_w_per_filt = tf.reduce_max(feature_maps, axis=2)
        max_w_per_filt_relu = tf.nn.relu(max_w_per_filt)
        
        # Convert to list and extend results directly
        max_w_per_filt_relu_np = max_w_per_filt_relu.numpy().flatten().tolist()
        
        replacement_true_lst.extend([outcome_bin] * num_filters)
        max_w_filt_lst.extend(max_w_per_filt_relu_np)

    # filt_nums list should be extended `num_patients` times.
    filt_nums = filt_nums * num_patients

    return replacement_true_lst, max_w_filt_lst, filt_nums