# ---------------------------------------------------------------------------------------
# Author              : Arif Thayal
# Project name        : _03_Lookalike_Model
# Purpose             : Main python code to execute the modeling flow
# Last modified by    : Arif Thayal
# Last modified date  : 09/05/2019
# ---------------------------------------------------------------------------------------

# import the libraries
import os, sys, importlib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date

%matplotlib inline
pd.options.display.html.table_schema = True

project = '_03_Lookalike_proj'
sys.path.append('./'+project+'/modules/')

# define the directory variables
input_dir = os.path.join(project,'input_files')
output_dir = os.path.join(project,'output_files')

# import modules of this project
import mod_01_datapreparation as data_prep
import mod_02_dataexploration as data_explore
import mod_04_modeltraining as model_train
import mod_05_modelscoring as model_score
#importlib.reload(model_train)
#importlib.reload(model_score.feat_select)


def train_predict():
  """used for training and validating the model"""
  # function parameters
  model_name = 'stacking'
  exec_type = 'train'
  metric_select='roc_auc'
  multicoll_check=False
  resampling=True
  gridsearch=False
    
  # fetching 2016 labelled data (to train)
  # ---------------------------------------------------------------------------------------
  raw_input_data = pd.read_csv(input_dir+"/data_2016.csv", sep=';')
  raw_input_data.sample(6)

  # transforming labelled data
  labelled_data = data_prep.transform_step(raw_input_data, 'bought_highbrow_wines')

  # exploring labelled data
  data_explore.exploration_step(labelled_data)


  # train the model and saving the evaluation metrics
  # --- metric_select can be any of these => 'roc_auc', 'accuracy', 'log_loss'
  # --- model_name can be any of these => 'xgboost', 'stacking'
  (y_test, y_pred, 
   y_pred_prob) = model_train.model_training_step(labelled_data,
                    output_dir, metric_select, model_name,resampling, gridsearch, multicoll_check)

  model_train.model_evaluation(y_test, y_pred, y_pred_prob)

  
def score_model():
  """used for scoring the model"""
  # function parameters
  exec_type='score'
  model_name='stacking'
    
  # 2017 unlabelled data (to score)
  # ---------------------------------------------------------------------------------------
  topredict_data = pd.read_csv(input_dir+"/data_2017.csv", sep=';')
  topredict_data.sample(6)

  # add dummy label variable (to align with transformation rules)
  topredict_data['bought_highbrow_wines'] = 0.0

  # transforming scoring data
  unlabelled_data = data_prep.transform_step(topredict_data, 'bought_highbrow_wines')

  # run the scoring and save the output
  model_score.model_scoring_step(unlabelled_data,output_dir,model_name)

def recompute_predictions():
  """used for changing the probability threshold and recomputing results"""
  niteration = 2
  nthreshold = 0.6
  model_name = 'xgboost'
  model_train.recompute_predict_threshold(model_name, niteration, nthreshold, output_dir)

  
def main(exec_type='score'):
  """ main method to run train or score models. Default is score"""
  if   (exec_type=='train'): train_predict()
  elif (exec_type=='score'): score_model()
  elif (exec_type=='recompute'): recompute_predictions()
  

if __name__ == '__main__':
    main('score')  # 'score' to score the model, 'train' to train the model
  


  
  
# Iteration 1:
#Model accuracy: 0.967
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.78      0.43      0.56      2845
#
#   micro avg       0.97      0.97      0.97     58706
#   macro avg       0.88      0.71      0.77     58706
#weighted avg       0.96      0.97      0.96     58706
#
#[[55517   344]
# [ 1612  1233]]


# Iteration 2:
#Model accuracy: 0.967
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.78      0.43      0.56      2845
#
#   micro avg       0.97      0.97      0.97     58706
#   macro avg       0.88      0.71      0.77     58706
#weighted avg       0.96      0.97      0.96     58706
#
#[[55517   344]
# [ 1612  1233]]
#[[ 0.99384186  0.00615814]
# [ 0.56660808  0.43339192]]
#AUC Final:0.714


## Iteration 3:
#Model accuracy: 0.963
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.68      0.46      0.55      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.82      0.73      0.76     58706
#weighted avg       0.96      0.96      0.96     58706
##
#[[55231   630]
# [ 1532  1313]]
#[[ 0.98872201  0.01127799]
# [ 0.53848858  0.46151142]]
#AUC Final:0.893


## Iteration 4:
#Model accuracy: 0.963
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.67      0.48      0.56      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.82      0.73      0.77     58706
#weighted avg       0.96      0.96      0.96     58706
#
#[[55199   662]
# [ 1485  1360]]
#[[ 0.98814916  0.01185084]
# [ 0.52196837  0.47803163]]
#AUC Final:0.911

# XGBOOST iteration 1:
#Model accuracy: 0.960
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.62      0.47      0.53      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.80      0.73      0.76     58706
#weighted avg       0.96      0.96      0.96     58706
#
#[[55045   816]
# [ 1520  1325]]
#[[ 0.98539231  0.01460769]
# [ 0.53427065  0.46572935]]
#AUC Final:0.926

#best params:{'xgboost__colsample_bytree': 0.8, 'xgboost__gamma': 0.5, 'xgboost__learning_rate': 0.7, 'xgboost__max_depth': 5, 'xgboost__min_child_weight': 1.5, 'xgboost__n_estimators': 150, 'xgboost__reg_alpha': 0.75, 'xgboost__reg_lambda': 0.45, 'xgboost__scale_pos_weight': 0.5, 'xgboost__subsample': 0.9}
#best score:0.971002645159

# XGBOOST iteration 2:
#  Model accuracy: 0.961
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.63      0.46      0.53      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.80      0.72      0.75     58706
#weighted avg       0.96      0.96      0.96     58706
#
#[[55096   765]
# [ 1542  1303]]
#[[ 0.98630529  0.01369471]
# [ 0.54200351  0.45799649]]
#AUC Final:0.927


# XGBOOST threshold iteration 1:

#Model accuracy: 0.963
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.74      0.37      0.49      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.86      0.68      0.74     58706
#weighted avg       0.96      0.96      0.96     58706
#
#[[55502   359]
# [ 1797  1048]]
#[[ 0.99357333  0.00642667]
# [ 0.63163445  0.36836555]]
#AUC Final:0.925


# XGBOOST No oversampling iteration 1:

#Model accuracy: 0.961
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.97      0.99      0.98     55861
#           1       0.62      0.48      0.54      2845
#
#   micro avg       0.96      0.96      0.96     58706
#   macro avg       0.80      0.73      0.76     58706
#weighted avg       0.96      0.96      0.96     58706
#
#[[55039   822]
# [ 1481  1364]]
#[[ 0.9852849   0.0147151 ]
# [ 0.52056239  0.47943761]]
#AUC Final:0.929


# Random Forest iteration 1:
#Model accuracy: 0.785
#Classification report:
#              precision    recall  f1-score   support
#
#           0       0.98      0.79      0.87     55861
#           1       0.15      0.72      0.24      2845
#
#   micro avg       0.78      0.78      0.78     58706
#   macro avg       0.56      0.75      0.56     58706
#weighted avg       0.94      0.78      0.84     58706
#
#[[44028 11833]
# [  802  2043]]
#[[ 0.78817064  0.21182936]
# [ 0.28189807  0.71810193]]
#AUC Final:0.841


## ---------------------------------------------------------------------------------------