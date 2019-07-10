from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

#----------------------------------------------------------------------------------------
def gridsearch_validate (model_name, model, X_train,y_train):
  """
  Run the GridSearchCV validation to find the best parameters for different classifiers.
  
  Args:
    model_name (str):    rf (for RandomForest) or xgboost (for XGB)
    model (obj):         classifier object of the model
    X_train (dataframe): X dataset 
    y_train (dataframe): y dataset
  Return:
    gridCV.best_params_
  
  """

  print("GridSearch started_Timestamp:{}".format(format(str(datetime.now()))))

  # GridSearch for tuning parameters
  if (model_name == 'xgboost'):
    param_combination = {model_name+'__n_estimators':[100, 150],
                         model_name+'__colsample_bytree':[0.6, 0.8],
                         model_name+'__gamma':[0.1, 0.5],
                         model_name+'__min_child_weight':[1.5, 3],
                         model_name+'__learning_rate':[0.3, 0.7],
                         model_name+'__max_depth':[3, 5],
                         model_name+'__subsample':[0.6, 0.9],
                         model_name+'__scale_pos_weight':[0.25, 0.75]
                        }    
    
  elif (model_name == 'rf'):
    param_combination = {model_name+'__n_estimators': [50, 100],
                         model_name+'__max_features': ['log2', 'sqrt','auto'], 
                         model_name+'__max_depth': [3, 5], 
                         model_name+'__min_samples_split': [0.9, 0.5],
                         model_name+'__min_samples_leaf': [5, 8]
                        }
    
  elif (model_name == 'svc'):
    param_combination = {model_name+'__kernel': ['linear', 'rbf'],
                         model_name+'__C': [1, 0.5, 0.025]
                        }
    
  elif (model_name == 'knn'):
    param_combination = {model_name+'__n_neighbors': [10, 15, 30, 40],
                         model_name+'__p': [2],
                         model_name+'__weights': ['uniform', 'distance']
                        }
  
  acc_scorer = make_scorer(accuracy_score)
  gridCV = GridSearchCV(estimator = model,
                        param_grid = param_combination,
                        cv=5, n_jobs=4,
                        iid=False, verbose=10,
                        scoring=acc_scorer)
  
  gridCV.fit(X_train,y_train)
  #print (gridCV.grid_scores_)
  print('best params:'+str(gridCV.best_params_))
  print('best score:'+str(gridCV.best_score_))

  print("GridSearch ended_Timestamp:{}".format(format(str(datetime.now()))))
  
  return gridCV.best_params_
  # -------------------------------------------------------------------------

def kfold_validate(model, X_tr, y_tr, nkfold, metric_score):
  
  # apply K-fold cross validation sets
  skf = StratifiedKFold(n_splits=nkfold, random_state=0, shuffle=False)

  # score
  scores = cross_val_score(model, X_tr, y_tr, scoring=metric_score,
                           n_jobs=-1, cv=skf)
  print("cross-validated score: "+str(np.mean(scores)))
  
  return (model)

