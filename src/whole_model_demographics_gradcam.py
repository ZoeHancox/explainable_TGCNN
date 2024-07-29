import numpy as np
import math
import pandas as pd
import random
from tensorflow import keras
import tensorflow as tf
from src import TGCNN_layer


class TGCNN_Model(tf.keras.Model):
    
    def __init__(self, num_filters=2, num_nodes=97, num_time_steps=100, 
                 filter_size=3, variable_gamma=True, exponential_scaling=True, dropout_rate = 0.2,
                lstm_units=16, fcl1_units = 128, LSTM_ablation=False, stride=1, activation_type="relu", no_timestamp=False,
                second_TGCNN_layer=True, num_labels=2):
        
        super(TGCNN_Model, self).__init__()
        # self.input_layer = tf.keras.Input(shape=(None, num_nodes, num_nodes, num_time_steps), sparse=True)
        self.LSTM_ablation = LSTM_ablation
        self.second_TGCNN_layer = second_TGCNN_layer
        
        self.tg_conv_layer1 = TGCNN_layer.TGCNN_layer(num_filters=num_filters, num_nodes=num_nodes, num_time_steps=num_time_steps, 
                                         filter_size=filter_size, variable_gamma=variable_gamma, exponential_scaling=exponential_scaling,
                                         stride=stride, no_timestamp=no_timestamp, name='tg_conv_layer1') #stride is probably>2 so this is the shorter stream 
        if second_TGCNN_layer==True:
            self.tg_conv_layer2 = TGCNN_layer.TGCNN_layer(num_filters=num_filters, num_nodes=num_nodes, num_time_steps=num_time_steps, 
                                         filter_size=filter_size, variable_gamma=variable_gamma, exponential_scaling=exponential_scaling,
                                         stride=2, no_timestamp=no_timestamp, name='tg_conv_layer2')
        
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        
        if activation_type == "relu" or "gelu":
            self.activation = tf.keras.layers.Activation(activation_type, name=activation_type)
        elif activation_type == "LeakyReLU":
            self.activation = tf.keras.layers.LeakyReLU(name='LeakyReLU')

        self.dy_du_branch1 = None
        self.dy_du_branch2 = None

        self.f_map_branch1 = None
        
        self.lstm = tf.keras.layers.LSTM(units=lstm_units) # input shape must be 3D with shape [batch,timesteps,feature]
        self.flatten = tf.keras.layers.Flatten()
        self.fcl1_short = tf.keras.layers.Dense(units=fcl1_units, activation=activation_type)
        self.fcl1_long = tf.keras.layers.Dense(units=fcl1_units, activation=activation_type)
        
        self.fcl2_short = tf.keras.layers.Dense(units=256, activation=activation_type)
        self.fcl2_long = tf.keras.layers.Dense(units=256, activation=activation_type)
        
        self.fcl3_short = tf.keras.layers.Dense(units=256, activation=activation_type)

        self.fcl_after_concat = tf.keras.layers.Dense(units=512)
        self.fcl_after_concat2 = tf.keras.layers.Dense(units=512)
        self.fcl_to_out = tf.keras.layers.Dense(units=num_labels)#, activation="softmax") softmax activation can't be used here with softmax_cross_entropy_with_logits
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.saved_layer_output = None
        
    def flat_to_out(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fcl1_short(x)
        x = self.dropout(x)
        out = self.fcl2_short(x)
        return out
        
    def call(self, inputs, demos):
#         print("inputs:", inputs)
#         print("demos", demos)
        if self.second_TGCNN_layer == False:
            x = self.tg_conv_layer1(inputs)
            #print("\n\nAfter 3DCNN layer out:", x)
            self.saved_layer_output=x
            #print("Feature maps from 3DCNN Layer", x)
            x = self.batchnorm1(x)
            #print("After batchnorm layer out:", x)
            x = self.activation(x)
            cnn_out = tf.squeeze(x, axis=2)
            #print("Just before LSTM:", x)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(cnn_out)

                #if self.LSTM_ablation == False: # with LSTM layer
                out = tf.transpose(cnn_out, perm=[0,2,1]) # switch axis 1 and 2 for LSTM input
                out = self.lstm(out)
                out = self.flat_to_out(out)
                #out = self.fcl_to_out(out)

                out = tf.keras.layers.Concatenate()([out, demos])
                out = self.fcl_after_concat(out)
                out = self.activation(out)
                out = self.fcl_after_concat2(out)
                out = self.activation(out)
                out = self.fcl_to_out(out)
            
            self.dy_du_branch1 = tape.gradient(out, cnn_out)
            #print(dy_du)

            
        elif self.second_TGCNN_layer == True:
            
            # long stream (stride = 1)#####################
            #print("input shape into 3D CNN", inputs.shape)
            x_long = self.tg_conv_layer1(inputs)
            #print("Output shape of 3DCNN:", x_long.shape)
            #print("\n\nAfter 3DCNN layer out:", x_long)
            self.saved_layer_output=x_long
            x_long = self.batchnorm1(x_long)
            x_long = self.activation(x_long)
            cnn_out1 = tf.squeeze(x_long, axis=2)
            self.f_map_branch1 = cnn_out1 # get the feature maps out
            
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(cnn_out1)

                #if self.LSTM_ablation == False: # with LSTM layer
                x_long = tf.transpose(cnn_out1, perm=[0,2,1]) # switch axis 1 and 2 for LSTM input
                x_long = self.lstm(x_long)
                # print("shape of LSTM output", x_long.shape)
                x_long = self.flatten(x_long)
                x_long = self.dropout(x_long)
                x_long = self.fcl1_long(x_long)
                x_long = self.dropout(x_long)
                out_long = self.fcl2_long(x_long)
                
           
                # short stream (stride = 2)#####################
                x_short = self.tg_conv_layer2(inputs)
                x_short = self.batchnorm2(x_short)
                x_short = self.activation(x_short)
                cnn_out2 = tf.squeeze(x_short, axis=2)

            # with tf.GradientTape(persistent=True) as tape2:
            #     tape2.watch(cnn_out2)
                
                # if self.LSTM_ablation == False: # with LSTM layer
                out_short = tf.transpose(cnn_out2, perm=[0,2,1]) # switch axis 1 and 2 for LSTM input
                out_short = self.lstm(out_short)
                out_short = self.flat_to_out(out_short)
            
             
            
                # concatenation of the two channels/streams
                out = tf.keras.layers.Concatenate()([out_long, out_short, demos])
                out = self.fcl_after_concat(out)
                out = self.activation(out)
                out = self.fcl_after_concat2(out)
                out = self.activation(out)
                out = self.fcl_to_out(out)

            self.dy_du_branch1 = tape1.gradient(out, cnn_out1)
            self.dy_du_branch2 = tape1.gradient(out, cnn_out2)
            
            
        return out
    
    def save_layer_output(self, file_path):
        if self.saved_layer_output is not None:
            np.save(file_path, self.saved_layer_output.numpy())
            print("Feature Maps Saved")
        else:
            print("Layer output not available")

