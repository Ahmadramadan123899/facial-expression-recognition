#! /usr/bin/python3
# Author : Kevin Feghoul

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import *
from sklearn.metrics import confusion_matrix

from utils import *

def save_results(history:Dict, id:int, ground_truth:np.ndarray, preds:np.ndarray, results_path:str) -> np.ndarray:
    
    cm = confusion_matrix(ground_truth, preds)

    results_visualization = os.path.join(results_path, 'plots')

    make_dirs(results_visualization)
    
    Num_epoch = [i for i in range(len(history['loss']))]

    # Paper protocol has no validation set, so only train metrics are plotted.
    # val_acc and val_loss have been removed from history.

    ### Graphe de l'accuracy en train uniquement
    plt.figure()
    plt.plot(Num_epoch, history['acc'], label="Train accuracy")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy over epochs")
    plt.grid(True)
    plt.savefig(os.path.join(results_visualization, str(id) + '_accuracy'))
    plt.close()

    ### Graphe de loss en train uniquement
    plt.figure()
    plt.plot(Num_epoch, history['loss'], label="Train loss")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss over epochs")
    plt.grid(True)
    plt.savefig(os.path.join(results_visualization, str(id) + '_loss'))
    plt.close()

    ### Graphe tps moyen de chaque batch par epoch
    plt.figure()
    plt.plot(Num_epoch, history['tps_par_moy_batch_par_epoch'], label="Temps moyen par batch")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Temps moyen par batch")
    plt.title("Mean time per batch over epochs")
    plt.grid(True)
    plt.savefig(os.path.join(results_visualization, str(id) + '_tps_par_batch'))
    plt.close()

    ### Graphe tps moyen par epoch
    plt.figure()
    plt.plot(Num_epoch, history['tps_par_epoch'], label="Temps par epoch")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Temps")
    plt.title("Time per epoch")
    plt.grid(True)
    plt.savefig(os.path.join(results_visualization, str(id) + '_tps_par_epoch'))
    plt.close()

    ### Matrice de confusion
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title("Confusion matrix (fold {})".format(id))
    plt.savefig(os.path.join(results_visualization, str(id) + '_confusion_matrix'))
    plt.close()

    return cm
    

def results_all(all_acc:List[float], all_f1:List[float], all_precision:List[float], all_recall:List[float], model_name:str, save_results_path:str):

    avg_acc, avg_precision, avg_recall = round(np.mean(all_acc), ndigits=2), round(np.mean(all_precision), ndigits=2), round(np.mean(all_recall), ndigits=2)
    std_acc, std_precision, std_recall  = round(np.std(all_acc), ndigits=2), round(np.std(all_precision), ndigits=2), round(np.std(all_recall), ndigits=2)

    print('\n\nAverage accuracy : {}, Std accuracy : {}'.format(avg_acc, std_acc))
    print('Average precision : {}, Std precision : {}'.format(avg_precision, std_precision))
    print('Average recall : {}, Std recall : {}'.format(avg_recall, std_recall))

    results = [avg_acc, avg_precision]

    barplot = sns.barplot(x=['Average_acc'], y=results)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results[xtick], results[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_results_path, model_name + '_scores'))
    plt.close()



def save_results_all_folds(all_results:List[Any], model_name:str, save_results_path:str, all_results_path:str, best_results_path:str, 
                           nb_frames_par_video:int,  
                           nb_landmarks_par_frame:int, nb_train:int, nb_val:int, nb_test:int) -> None:

    all_acc, all_prec, all_rec, all_cm, all_acc_final, all_cm_final = all_results
    
    # all_acc == all_acc_final since there's no separate val set under the
    # paper protocol; both are the test fold accuracy at end of training.
    avg_acc      = round(np.mean(all_acc),       ndigits=2)
    avg_prec     = round(np.mean(all_prec),      ndigits=2)
    avg_rec      = round(np.mean(all_rec),       ndigits=2)
    std_acc      = round(np.std(all_acc),        ndigits=2)
    std_prec     = round(np.std(all_prec),       ndigits=2)
    std_rec      = round(np.std(all_rec),        ndigits=2)
    avg_acc_test = round(np.mean(all_acc_final), ndigits=2)
    std_acc_test = round(np.std(all_acc_final),  ndigits=2)

    print('Average accuracy  : {}, Std accuracy  : {}'.format(avg_acc,  std_acc))
    print('Average precision : {}, Std precision : {}'.format(avg_prec, std_prec))
    print('Average recall    : {}, Std recall    : {}'.format(avg_rec,  std_rec))

    ### Stockage des scores
    scores = [avg_acc_test, std_acc_test, avg_acc, std_acc,
              model_name, nb_frames_par_video, nb_landmarks_par_frame,
              nb_train, nb_val, nb_test, save_results_path]
    append_new_line(all_results_path, ', '.join('{}'.format(k) for k in scores))

    data = pd.read_csv(all_results_path, sep=",", header=None)
    data.columns = ['acc_test', 'std_acc_test', 'acc_val', 'std_acc_val',
                    'model_name', 'nb_frames_par_video', 'nb_landmarks_par_frame',
                    'nb_train', 'nb_val', 'nb_test', 'path']
    data.sort_values('acc_test', ascending=False, inplace=True)
    data.to_csv(best_results_path, index=None, sep=',')

    ### Barplot accuracy par fold
    x_acc       = ['fold_' + str(i) for i in range(1, len(all_acc) + 1)] + ['avg_acc']
    x_precision = ['fold_' + str(i) for i in range(1, len(all_prec) + 1)] + ['avg_precision']
    x_recall    = ['fold_' + str(i) for i in range(1, len(all_rec) + 1)] + ['avg_recalls']

    results_acc  = all_acc  + [avg_acc]
    results_prec = all_prec + [avg_prec]
    results_rec  = all_rec  + [avg_rec]

    barplot = sns.barplot(x=x_acc, y=results_acc)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_acc[xtick], results_acc[xtick],
                     horizontalalignment='center', size='small', color='black', weight='semibold')
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_results_path, model_name + '_accuracy'))
    plt.close()

    barplot = sns.barplot(x=x_precision, y=results_prec)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_prec[xtick], results_prec[xtick],
                     horizontalalignment='center', size='small', color='black', weight='semibold')
    plt.ylabel("Precision")
    plt.savefig(os.path.join(save_results_path, model_name + '_precision'))
    plt.close()

    barplot = sns.barplot(x=x_recall, y=results_rec)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_rec[xtick], results_rec[xtick],
                     horizontalalignment='center', size='small', color='black', weight='semibold')
    plt.ylabel("Recall")
    plt.savefig(os.path.join(save_results_path, model_name + '_recall'))
    plt.close()

    ### Matrice de confusion cumulée sur tous les folds
    sum_cm_final = np.zeros(all_cm_final[0].shape)
    for cm in all_cm_final:
        sum_cm_final += cm

    plt.figure(figsize=(7, 5))
    sns.heatmap(sum_cm_final.astype('int64'), annot=True, fmt='d', cbar=False, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title("Cumulative confusion matrix — all folds (test)")
    plt.savefig(os.path.join(save_results_path, 'sum_confusion_matrix_final_test'))
    plt.close()
