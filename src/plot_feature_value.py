import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ave_grad_all_f_maps(grads: tf.Tensor, title:str='Average Gradient of All Feature Maps', ylab:str='Normalised Mean Gradient Value', filename:str=None):
    """Plot the average gradient of all feature maps.

    Args:
        grads (tf.array): gradients of the CNN layer output in respect to 
                         the model output.
        filename (str): filename pre plot type.

    Return:
        np.array: normalised features from the CNN layer.
    """
    squeezed = tf.squeeze(grads)
    squeezed = np.array(squeezed)
    mean_features = np.mean(squeezed, axis=0)

    grads_min = np.min(mean_features)
    grads_max = np.max(mean_features)
    norm_features = (mean_features - grads_min) / (grads_max - grads_min)
    #norm_features = mean_features/np.sum(mean_features)

    plt.plot(norm_features)
    plt.title(title)
    plt.xlabel("Feature Map Index")
    plt.ylabel(ylab)
    if filename != None:
        plt.savefig(f"{filename}_ave_all_f_maps.png")
    plt.show()
    return norm_features

def plot_indv_grad_f_maps(grads: tf.Tensor, title:str='Gradient of Each Feature Map', ylab:str='Normalised Mean Gradient Value', filename:str=None):

    plt.figure(figsize=(12, 10))
    for map in range(grads.shape[1]):
        np_grads = np.array(grads[0, map, :])
        grads_min = np.min(np_grads)
        grads_max = np.max(np_grads)
        norm_grads = (np_grads - grads_min) / (grads_max - grads_min)

        plt.plot(norm_grads, label=f'Feature Map {map}')

    plt.title(title)
    plt.xlabel("Feature Map Index")
    plt.ylabel(ylab)
    if filename != None:
        plt.savefig(f"{filename}_ave_all_f_maps.png")
    plt.legend(loc='upper left')
    plt.show()

def violin_plots(filt_nums:list, max_w_filt_lst:list, replacement_true_lst:list):
    """Create a seaborn violin plot showing which filter had the highest variance in 
    activation for each patient. 

    Args:
        filt_nums (list): list of filter numbers.
        max_w_filt_lst (list): maximum weight from each feature map.
        replacement_true_lst (lst): class/outcome.
    """
    plt.figure(figsize=(14,6))

    palette = sns.color_palette("colorblind")
    sns.violinplot(x=filt_nums, y=max_w_filt_lst, hue=replacement_true_lst, split=True, inner="quartile", palette=palette[1:3], cut=0.5)

    plt.legend(title='Hip Replacement')
    plt.xlabel('Filter Number')
    plt.ylabel('Maximum Activation in Feature Map')
    plt.xlim(-1,15.5)
    # Calculate the IQR
    q1 = np.percentile(max_w_filt_lst, 25)
    q3 = np.percentile(max_w_filt_lst, 75)
    iqr = q3 - q1
    lower_bound = max(min(max_w_filt_lst), q1 - 1.5 * iqr)
    upper_bound = min(max(max_w_filt_lst), q3 + 1.5 * iqr)
    plt.ylim([lower_bound - 10 * iqr, upper_bound + 10 * iqr])
    plt.savefig("feature_map_plots/violin_plots1.png", bbox_inches='tight')
    plt.show()

    if max(filt_nums) > 16:
        plt.figure(figsize=(14,6))

        palette = sns.color_palette("colorblind")
        sns.violinplot(x=filt_nums, y=max_w_filt_lst, hue=replacement_true_lst, split=True, inner="quartile", palette=palette[1:3], cut=0.5)

        plt.legend(title='Hip Replacement')
        plt.xlabel('Filter Number')
        plt.ylabel('Maximum Activation in Feature Map')
        plt.xlim(15.5,31.5)
        # Calculate the IQR
        q1 = np.percentile(max_w_filt_lst, 25)
        q3 = np.percentile(max_w_filt_lst, 75)
        iqr = q3 - q1
        lower_bound = max(min(max_w_filt_lst), q1 - 1.5 * iqr)
        upper_bound = min(max(max_w_filt_lst), q3 + 1.5 * iqr)
        plt.ylim([lower_bound - 10 * iqr, upper_bound + 10 * iqr])
        plt.savefig("feature_map_plots/violin_plots2.png", bbox_inches='tight')
        plt.show()


def max_act_box_plots(filt_nums:list, max_w_filt_lst:list, replacement_true_lst:list):
    """Create a seaborn boxplot plot with pairs for each filter showing the distribution 
    of the variance for each class over the patients.

    Args:
        filt_nums (list): list of filter numbers.
        max_w_filt_lst (list): maximum weight from each feature map.
        replacement_true_lst (lst): class/outcome.
    """
    plt.figure(figsize=(14,6))

    palette = sns.color_palette("colorblind")
    sns.boxplot(x=filt_nums, y=max_w_filt_lst, hue=replacement_true_lst, palette=palette[1:3], fliersize=0)

    for x_tick in range(1, max(filt_nums)):
        plt.axvline(x=x_tick-0.5, color='grey', linestyle='--', linewidth=0.5)

    plt.legend(title='Hip Replacement')
    plt.xlabel('Filter Number')
    plt.ylabel('Maximum Activation in Feature Map')
    # Calculate the IQR
    q1 = np.percentile(max_w_filt_lst, 25)
    q3 = np.percentile(max_w_filt_lst, 75)
    iqr = q3 - q1
    
    # Set y-axis limits to focus on the IQR with some padding
    lower_bound = max(min(max_w_filt_lst), q1 - 1.5 * iqr)
    upper_bound = min(max(max_w_filt_lst), q3 + 1.5 * iqr)
    plt.ylim([lower_bound - 10 * iqr, upper_bound + 10 * iqr])
    # plt.xlim(-1,15.5)
    sns.despine(trim=True)
    plt.savefig("feature_map_plots/box_plots.png", bbox_inches='tight')
    plt.show()