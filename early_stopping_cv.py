import numpy as np
from src import utils, trnvaltst_sigmoid_oned, plot_figures
import tensorflow as tf


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_made = False
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

             
        
    def __call__(self, val_loss, train_batch_losses, val_batch_loss, train_cal_slope, val_cal_slope, train_batch_acc, val_batch_acc, 
                 train_batch_auc, val_batch_auc, train_batch_prec, val_batch_prec, train_batch_recall, val_batch_recall,  train_batch_f1, 
                 val_batch_f1, model, run_name, batched_graphs_holdout, batched_labels_holdout, batched_holdout_indices, batched_graphs_holdout2, 
                 batched_labels_holdout2, batched_holdout2_indices, reg_strength, class_weights, L1_ablation, L2_ablation, graph_reg_strength, 
                 graph_reg_incl, weighted_loss, improved_acc, test_patients, recal_test_patients, demo, save_filters):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, run_name, batched_graphs_holdout, batched_labels_holdout, batched_holdout_indices,batched_graphs_holdout2, 
                                 batched_labels_holdout2, batched_holdout2_indices, reg_strength, class_weights, L1_ablation, L2_ablation, graph_reg_strength, 
                                 graph_reg_incl, weighted_loss, improved_acc, test_patients, recal_test_patients, demo, save_filters)
            self.checkpoint_made = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
            self.checkpoint_made = False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, run_name, batched_graphs_holdout, batched_labels_holdout, batched_holdout_indices, batched_graphs_holdout2, 
                                 batched_labels_holdout2, batched_holdout2_indices, reg_strength, class_weights, L1_ablation, L2_ablation, graph_reg_strength, 
                                 graph_reg_incl, weighted_loss, improved_acc, test_patients, recal_test_patients, demo, save_filters)
            self.counter = 0
            self.checkpoint_made =True
        
    def save_checkpoint(self, val_loss, model, run_name, batched_graphs_holdout, batched_labels_holdout, batched_holdout_indices, batched_graphs_holdout2, 
                        batched_labels_holdout2, batched_holdout2_indices, reg_strength, class_weights, L1_ablation, L2_ablation, graph_reg_strength, 
                        graph_reg_incl, weighted_loss, improved_acc, test_patients, recal_test_patients, demo, save_filters):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss # replace the lowest loss value with the new lowest loss value
        if save_filters:
            with open(run_name+'_filter.npy', 'wb') as f:
                np.save(f, model.tg_conv_layer1.trainable_weights[0].numpy())
        # print("Feature maps from CNN saving...")
        # model.save_layer_output(f"{run_name}_CNN_feature_maps.npy")
        print("Model weights saving...")
        model.save_weights(f"{run_name}_CNN_layer")
        
        
        if improved_acc:
            test_logits_list, test_demo_list, test_pat_num_list = [], [], []
            test_loss_list, test_acc_list, test_prec_list, test_recall_list, test_auc_list, test_f1_list = [], [], [], [], [], []
            for x_batch_test, y_batch_test, holdout_indices_list in zip(batched_graphs_holdout, batched_labels_holdout, batched_holdout_indices):
                
                holdout_demo_vals, holdout_demo_list, holdout_pat_num = utils.convert_demos_to_tensor(test_patients, holdout_indices_list, demo)
                test_demo_list = test_demo_list + holdout_demo_list
                test_pat_num_list = test_pat_num_list + holdout_pat_num
                
                #print("test_demo_list", test_demo_list)
                
                if demo == False:
                    holdout_demo_vals = None
                
                test_logits, test_loss, test_acc, test_prec, test_recall, test_auc, test_f1, \
                model = trnvaltst_sigmoid_oned.val_step(x_batch_test, y_batch_test, holdout_demo_vals, reg_strength,
                                            class_weights, model, L1_ablation, L2_ablation,
                                             graph_reg_strength, graph_reg_incl, weighted_loss, demo)
                
               
                test_logits_list.append(test_logits)                
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                test_prec_list.append(test_prec)
                test_recall_list.append(test_recall)
                test_auc_list.append(test_auc)
                test_f1_list.append(test_f1)
                
            
            
            test_logits = tf.concat(test_logits_list, axis=0)
            demo_list = tf.concat(test_demo_list, axis=0)
            pat_num_list = tf.concat(test_pat_num_list, axis=0)

            file_full_name_proba = 'pred_proba_and_true/'+run_name+'_holdout1_proba.npy'
            with open(file_full_name_proba, 'wb') as f:
                np.save(f, test_logits)
                
             
            file_full_name_demo = 'pred_proba_and_true/'+run_name+'_holdout1_demo.npy'
            with open(file_full_name_demo, 'wb') as f:
                np.save(f, demo_list)    

            file_full_name_pat_num = 'pred_proba_and_true/'+run_name+'_holdout1_pat_num.npy'
            with open(file_full_name_pat_num, 'wb') as f:
                np.save(f, pat_num_list) 
                
            print("\nTEST/HOLDOUT METRICS:")
            print(f"Test AUC scores: {np.mean(test_auc_list):.4f}")
            print(f"Test accuracy: {np.mean(test_acc_list) :.4%}")
            print(f"Test Recall: {np.mean(test_recall_list):.4f}")
            print(f"Test Precision: {np.mean(test_prec_list):.4f}")
            print(f"Test F1: {np.mean(test_f1_list):.4f}")

            test_logits_list = test_logits.numpy().tolist()
            
            flattened_test_labels = [int(item) for sublist in batched_labels_holdout for array in sublist for item in array]
            test_cal_slope = utils.calibration_slope(flattened_test_labels, test_logits_list)
            print(f"Test calibration slope: {test_cal_slope:.4f}")
            

            # Checking the probability distributions and outcome distribution on the final batch   
            # plot_figures.draw_confusion_mat(y_batch_test, test_logits, ['none','hip'], run_name=None, ran_search_num=2222, data_type="T")
            #plot_figures.draw_calibration_curve(np.array(flattened_test_labels), np.array(test_logits_list), run_name=None, ran_search_num=2222)
            
            print("*"*40)
            
            
            ## STUFF FOR THE RECALIBRATION TEST SET (TEST SET 2)
            test_demo_list2, test_logits_list2, test_pat_num_list2 = [], [], []
            for x_batch_test2, y_batch_test2, holdout2_indices_list in zip(batched_graphs_holdout2, batched_labels_holdout2, batched_holdout2_indices):
                
                holdout2_demo_vals, holdout2_demo_list, holdout2_pat_num = utils.convert_demos_to_tensor(recal_test_patients, holdout2_indices_list, demo)
                test_demo_list2 = test_demo_list2 + holdout2_demo_list
                test_pat_num_list2 = test_pat_num_list2 + holdout2_pat_num
                
                if demo == False:
                    holdout2_demo_vals = None
                
                test2_logits, test2_loss, test2_acc, test2_prec, test2_recall, test2_auc, test2_f1, \
                model = trnvaltst_sigmoid_oned.val_step(x_batch_test2, y_batch_test2, holdout2_demo_vals, reg_strength,
                                            class_weights, model, L1_ablation, L2_ablation,
                                             graph_reg_strength, graph_reg_incl, weighted_loss, demo)                               

                test_logits_list2.append(test2_logits)
                
                
            test2_logits = tf.concat(test_logits_list2, axis=0)
            demo_list2 = tf.concat(test_demo_list2, axis=0)
            pat_num_list2 = tf.concat(test_pat_num_list2, axis=0)

            file_full_name_proba = 'pred_proba_and_true/'+run_name+'_holdout2_proba.npy'
            with open(file_full_name_proba, 'wb') as f:
                np.save(f, test2_logits)
        
            file_full_name_demo = 'pred_proba_and_true/'+run_name+'_holdout2_demo.npy'
            with open(file_full_name_demo, 'wb') as f:
                np.save(f, demo_list2)  

            file_full_name_pat_num = 'pred_proba_and_true/'+run_name+'_holdout2_pat_num.npy'
            with open(file_full_name_pat_num, 'wb') as f:
                np.save(f, pat_num_list2) 
                
             
        
        
        
    def print_checkpoint_metric(self, train_batch_losses, val_batch_loss, train_cal_slope, val_cal_slope, train_batch_acc, val_batch_acc, 
                                train_batch_auc, val_batch_auc, train_batch_prec, val_batch_prec, train_batch_recall, val_batch_recall, 
                                train_batch_f1, val_batch_f1):
        
        checkpoint_train_loss = np.mean(train_batch_losses)
        checkpoint_val_loss = np.mean(val_batch_loss)
        
        checkpoint_train_cal_slope = train_cal_slope
        checkpoint_val_cal_slope = val_cal_slope
        
        checkpoint_train_acc = np.mean(train_batch_acc)
        checkpoint_val_acc = np.mean(val_batch_acc)
        
        checkpoint_train_auc = np.mean(train_batch_auc)
        checkpoint_val_auc = np.mean(val_batch_auc)
        
        checkpoint_train_prec = np.mean(train_batch_prec)
        checkpoint_val_prec = np.mean(val_batch_prec)
        
        checkpoint_train_recall = np.mean(train_batch_recall)
        checkpoint_val_recall = np.mean(val_batch_recall)
        
        checkpoint_train_f1 = np.mean(train_batch_f1)
        checkpoint_val_f1 = np.mean(val_batch_f1)
        
        return checkpoint_train_loss, checkpoint_val_loss, checkpoint_train_cal_slope, checkpoint_val_cal_slope, checkpoint_train_acc, checkpoint_val_acc, checkpoint_train_auc, checkpoint_val_auc, checkpoint_train_prec, checkpoint_val_prec, checkpoint_train_recall, checkpoint_val_recall, checkpoint_train_f1, checkpoint_val_f1

