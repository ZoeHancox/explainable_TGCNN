import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_ave_grad_all_f_maps(grads: tf.Tensor):
    """Plot the average gradient of all feature maps.

    Args:
        grads (tf.array): gradients of the CNN layer output in respect to 
                         the model output.

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
    plt.title("Average Gradient of All Feature Maps")
    plt.xlabel("Feature Map Length")
    plt.ylabel("Normalised Mean Gradient Value")
    plt.show()
    return norm_features

def plot_indv_grad_f_maps(grads: tf.Tensor):

    plt.figure(figsize=(12, 10))
    for map in range(grads.shape[1]):
        np_grads = np.array(grads[0, map, :])
        grads_min = np.min(np_grads)
        grads_max = np.max(np_grads)
        norm_grads = (np_grads - grads_min) / (grads_max - grads_min)

        plt.plot(norm_grads, label=f'Feature Map {map}')

    plt.title("Gradient of Each Feature Map")
    plt.xlabel("Feature Map Length")
    plt.ylabel("Normalized Mean Gradient Value")
    plt.legend(loc='upper left')  # This will automatically use the labels specified in plt.plot
    plt.show()