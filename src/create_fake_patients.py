import pandas as pd
import random
import tensorflow as tf

def create_fake_index_list(max_events, max_nodes):
    """Create indices for fake patients e.g. [[56,23,99], [34,3,98]]
    such that each sublist would be [start_node, end_node, timestep].

    Returns:
        list of lists: indices for 3D matrices.
    """

    # minimum number of events needs to be 2 to make a graph
    min_num_events = 2

    indices_list_of_lists = []
    for i in range(random.randint(min_num_events, max_events)):
        if i!= 0:
            indices_list_of_lists.append([random.randint(0, max_nodes-1),indices_list_of_lists[i-1][0],max_events-i])
        else:
            indices_list_of_lists.append([random.randint(0, max_nodes-1),random.randint(0, max_nodes-1),max_events-i])
    
    return indices_list_of_lists


def create_fake_patient_df(num_patients, max_events, max_nodes):
    """Create df with columns: User number | Indices (int, list of lists) |
    Values (list of floats) | Num time steps (int) | gender (bin int)

    Args:
        num_patients (int): number of patient rows to generate
        max_events (int): maximum number of events/visits a 'patient' can have
        max_nodes (int): maximum number of different types of nodes/read codes that can be used

    Returns:
        dataframe: df with columns for inputs and labels
    """
    # create a dictionary with the index as keys and the values as lists
    data = {'user': [i for i in range(1, num_patients)],
            'indices': [create_fake_index_list(max_events, max_nodes) for i in range(1, num_patients)],
            'values':0,
            'num_time_steps':0,
            'gender':0,
            'imd_quin':0,
            'age_at_label_event':0,
            'replace_type': 'n',
            'indices_len':0
            }

    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    for row_num, row in df.iterrows():
        indices_list = df.iloc[row_num,1] 
        num_timesteps = len([elem for elem in indices_list])
        df.iloc[row_num, 3] = num_timesteps

    values_list = []
    gender_list = []
    imd_list = []
    age_list = []
    replace_list = []
    for row_num, row in df.iterrows():
        num_timesteps = df.iloc[row_num,3]
        values_list.append([random.uniform(0.0, 1.0) for i in range(num_timesteps)])
        gender_list.append(random.randint(0, 1))
        imd_list.append(float(random.randint(0, 4))+1.0)
        age_list.append(float(random.randint(39, 89))+1.0)
        replace_list.append(random.choice(['hip', 'none']))

    df['values'] = values_list
    df['gender'] = gender_list
    df['imd_quin'] = imd_list
    df['age_at_label_event'] = age_list
    df['replace_type'] = replace_list
    df['indices_len'] = df['indices'].apply(lambda x: len(x))

    return df




def create_fake_int_label(num_patients):
    """ Create a dataframe with one column randomly selecting 0 to 2 of size num_patients,
    to represent patient label.

    Args:
        num_patients (int): Number of patients (needs to be the same as the input number)

    Returns:
        dataframe: One column dataframe with an integer between 0 and 3.
    """
    
    # create a list of random integers between 0 and 2 (inclusive)
    random_values = [random.randint(0, 2) for i in range(num_patients)]

    # create a Pandas DataFrame with one column containing the random values
    df = pd.DataFrame({'int_label': random_values})

    return df

def return_fake_pat(num_patients, max_visits, max_nodes, hip_or_knee, n):
    """_summary_

    Args:
        num_patients (int): number of patients to generate.
        max_visits (int): maximum number of timesteps to include.
        max_nodes (_type_): maximum number of Read Codes to include.
        hip_or_knee (str): which replacement type is being predicted.
        n (int): patient number to plot.

    Returns:
        tf.sparse array: Sparse patient graph 3D.
        tf.sparse array: Sparse patient graph 4D.
        tf array: demographics including gender, IMD quin and a placeholder.
                    for age as you can't normalise when only one value.
        str: 'hip' or 'knee'.
        binary: 1 for replacement, 0 for no replacement.

    """
    if n >= num_patients:
        raise ValueError(f"Input 'n' must be smaller than input 'num_patients'. Received n={n}, num_patients={num_patients}.")
    
    cv_patients = create_fake_patient_df(num_patients=num_patients, max_events=max_visits, max_nodes=max_nodes)
    i_list = cv_patients.iloc[n]['indices'] # indices from patient cell
    v_list = cv_patients.iloc[n]['values'] # values from patient cell

    individual_sparse = tf.sparse.SparseTensor(i_list, v_list, (max_nodes, max_nodes, 100))

    # Adding the sparse tensor to a list of all the tensors
    ordered_indiv = tf.sparse.reorder(individual_sparse) # reorder required for tensor to work (no effect to outcome)
 
    # expand the dims to have a batch size for the model
    input_4d = tf.sparse.expand_dims(ordered_indiv,axis=0)

    outcome = cv_patients.iloc[n]['replace_type']

    def classify_outcome(outcome):
        return 1 if outcome == hip_or_knee else 0

    outcome_bin = classify_outcome(outcome)

    demos = cv_patients[['gender', 'imd_quin', 'age_at_label_event']].iloc[n]
    demos_z = demos.copy()
    demos_z['age_zscore'] = 2
    demos_z = demos_z.apply(pd.to_numeric)  
    demo_vals = demos_z[['gender', 'imd_quin', 'age_zscore']].values 
    demo_tensor = tf.convert_to_tensor([demo_vals])

    return ordered_indiv, input_4d, demo_tensor, outcome, outcome_bin
