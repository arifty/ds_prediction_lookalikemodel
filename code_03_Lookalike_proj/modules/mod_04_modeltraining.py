import pandas as pd
import numpy as np
import seaborn as sns
import operator
from matplotlib import pyplot as plt
from matplotlib import pylab
from datetime import datetime, timedelta, date
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.metrics import roc_curve, precision_recall_curve
from vecstack import stacking, StackingTransformer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, InstanceHardnessThreshold


import mod_03_featureselection as feat_select
import fn_data_io as data_io
import fn_gridsearch_validate as grd_search


SEED = 42 # for reproducibility
NFOLD = 5 # for stacking out-of-fold predictions


# Class to extend the Sklearn classifier
class sklearn_helper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


# Training the model - Classification
#----------------------------------------------------------------------------------------
def model_training_step(data_totrain, path,
                        metric_select='accuracy', model_name = 'xgboost', 
                        resampling=False, gridsearch=False, multicoll_check=False):
  """Calling all other steps for model training and scoring
     1: train the model
     2: evaluate the model with the classification metrics
     3: save the model for further use for scoring
  """ 
  # logging
  print("Model training started_Timestamp:{}".format(format(str(datetime.now()))))
  
  if (metric_select=='accuracy'): metric_score = accuracy_score
  elif (metric_select=='roc_auc'): metric_score = roc_auc_score
  elif (metric_select=='log_loss'): metric_score = log_loss
  
  # splitting features and train
  customerid_train = data_totrain['masked_customer_id']
  features_data = data_totrain.set_index('masked_customer_id')  
  labels = features_data['bought_highbrow_wines'].astype(int)
  features = features_data[features_data.columns.difference(['bought_highbrow_wines'])]
  
  # encode the categorical features and scale
  features_scaled = feat_select.data_preprocessing(features)
  
  # get the train and test data splits
  (features_train, features_test, 
  label_train, label_test) = train_test_split(features_scaled, labels, test_size=.3, 
                                               random_state=SEED, stratify=labels)
  
  # feature selection procedure before training the data
  (featureslist, X_train, 
  y_train) = feat_select.feature_selection_step(features_train,label_train, 
                                                path, 'train', multicoll_check)
  
  #train the model
  model_pipe = model_train_clf (X_train, y_train, metric_score, model_name, 
                                           resampling, gridsearch)
  
  # prediction on test data
  (featureslist, X_test,
   y_test) = feat_select.feature_selection_step(features_test,label_test, 
                                                path, 'test', multicoll_check)

  y_pred = model_pipe.predict(X_test)  
  y_pred_prob = model_pipe.predict_proba(X_test)[:, 1]
  
  # saving the model step
  save_model_predict(model_pipe, model_name, y_test, y_pred, y_pred_prob, path)
  
  # logging
  print("Model training completed_Timestamp:{}".format(format(str(datetime.now()))))  
  
  return (y_test, y_pred, y_pred_prob)

# Classification model
#----------------------------------------------------------------------------------------
def model_train_clf (X, y, metric_score, model_name, resampling, gridsearch):
  """
  Run the Classification algorithm (different algos)  This function can be used to 
  train & test.
  Args:
    X (dataframe): predictor variables (input featrues data)
    y (series): target variable (variable to be predicted)
    model_name (str): By default uses 'xgboost'. Available params are:
                      'xboost', 'rf', 'svc', 'stacking' 
  Returns:
    model_pipe, y_train, y_test, y_pred, y_pred_prob
  """
  print('Before Sampling')
  print(sorted(Counter(y).items()))
  
  if (resampling):
    # get the over-sampled data, because of the class imbalance
    (X, y) = perform_over_sampling(X, y, samp_strat ='minority')
    
    print('After Sampling')
    print(sorted(Counter(y).items()))
    
  if (model_name == 'xgboost'):
    model_pipe = Pipeline([(model_name, XGBClassifier(
                                          n_estimators= 150,
                                          colsample_bytree=0.8,
                                          gamma=0.5,
                                          learning_rate=0.5,
                                          max_depth=5,
                                          min_child_weight=3,
                                          reg_alpha=0.75,
                                          reg_lambda=0.45,
                                          nthread=6, 
                                          scale_pos_weight=0.7,
                                          subsample=0.7,
                                          random_state=SEED))])
   
  elif (model_name == 'rf'):
    model_pipe = Pipeline([(model_name, RandomForestClassifier(
                                          n_jobs= 4,
                                          n_estimators= 50,
                                          warm_start= True, 
                                          max_features= 'log2',
                                          max_depth= 3,
                                          min_samples_split= .5,
                                          min_samples_leaf= 5,
                                          verbose= 0,
                                          random_state=SEED))])
  elif (model_name == 'knn'):
    model_pipe = Pipeline([(model_name, KNeighborsClassifier(
                                          n_neighbors= 10,
                                          weights= 'distance',
                                          p= 2,
                                          n_jobs=-1))])
  
  elif (model_name == 'svc'):
    model_pipe = Pipeline([(model_name, SVC(
                                          kernel= 'linear',
                                          gamma= 'scale',
                                          C= 0.5,
                                          verbose = True,
                                          random_state=SEED))])
   
  elif (model_name == 'stacking'):
    # construct the first level estimators for stacking
    estimators_L1 = [
      #('rf', RandomForestClassifier(
      #            n_jobs= -1,
      #            n_estimators= 50,
      #            warm_start= True, 
      #            max_features= 'log2',
      #            max_depth= 3,
      #            min_samples_split= .5,
      #            min_samples_leaf= 5,
      #            verbose= 0,
      #            random_state=SEED)),
      ('ada', AdaBoostClassifier(
                  n_estimators= 100,
                  learning_rate= 0.6,
                  random_state=SEED)),
      ('xgboost', XGBClassifier(
                  n_estimators= 150,
                  colsample_bytree=0.8,
                  gamma=0.5,
                  learning_rate=0.6,
                  max_depth=5,
                  min_child_weight=1.5,
                  reg_alpha=0.75,
                  reg_lambda=0.45,
                  nthread=6, 
                  scale_pos_weight=0.7,
                  subsample=0.7,
                  random_state=SEED))
      
#      ('knn', KNeighborsClassifier(
#                                  n_neighbors= 10,
#                                  weights= 'uniform',
#                                  p= 2,
#                                  n_jobs=-1))
      
#      ('svc', SVC(
#                  kernel= 'linear',
#                  gamma= 'scale',
#                  C= 0.5,
#                  random_state=SEED))
    ]

    # stacking with variant A and given metric_score
    model_pipe = prepare_stacking_pipe(estimators_L1, 'A', metric_score, NFOLD)
  
  
  # if set True, find the best params from the GridSearch output
  if (gridsearch):
    gridsearch_best_params = grd_search.gridsearch_validate(model_name, model_pipe, 
                                                          X, y)
    model_pipe.set_params(**gridsearch_best_params)
  
  # train fit model
  model_pipe = model_pipe.fit(X, y)
    
  return (model_pipe)



#Prepare the stack pipe
def prepare_stacking_pipe(l1_estim, variantAB, metric_score, nfold):
  """preparing the stcking pipe model for classification
     Meta-learner used is Logistic Regression        """
  # initialize stacking transformer
  stack_L1 = StackingTransformer(estimators=l1_estim,
                            regression=False,
                            variant=variantAB,
                            needs_proba=False,
                            metric=metric_score,
                            n_folds=nfold,
                            shuffle=True,
                            stratified=True,
                            random_state=SEED,
                            verbose=2)
  
  # final meta learner model
  meta_learner = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear', 
                              random_state=SEED)
    
  # creating stacking Pipeline
  stack_steps = [('stacking', stack_L1),
                 ('meta_model', meta_learner)]

  return Pipeline(stack_steps)


#Over-sampling: SMOTE
def perform_over_sampling(X_train, y_train, samp_strat = 'minority'):
  """For the imbalanced data, Over-sampling using SMOTE for the minority class"""
  
  sampler = SMOTE(sampling_strategy=samp_strat, random_state=SEED)
  #sampler = SMOTEENN(sampling_strategy=samp_strat, random_state=SEED)  
  #sampler = InstanceHardnessThreshold(sampling_strategy=samp_strat, random_state=SEED)  
  
  X_sm, y_sm = sampler.fit_resample(X_train, y_train)

  X_train_res = pd.DataFrame(X_sm, columns=X_train.columns.tolist())
  y_train_res = pd.Series(y_sm)
  
  return (X_train_res, y_train_res)


# evaluation
def model_evaluation(y_test, y_pred, y_pred_prob):
  """ Evaluating the model with classification metrics
      printing the scores and showing plots
  """
  print('Model accuracy: %.3f'% accuracy_score(y_test, y_pred))
  print('Classification report:')
  print(classification_report(y_test, y_pred))
  
  # plot confusion matrix
  conf_matrix_test = confusion_matrix(y_test, y_pred)
  classes = y_test.unique().tolist()
  plot_confusion_matrix(y_test, y_pred,classes,normalize=False)
  plot_confusion_matrix(y_test, y_pred,classes,normalize=True)
  
  # plot ROC curve with AUC
  fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)    
  auc_score = roc_auc_score(y_test, y_pred_prob)
  print('AUC Final:%.3f'% auc_score)
  plot_roc_curve(fpr, tpr, auc_score)
  
  plot_prec_recall_curve(y_test, y_pred_prob)
  
        
def save_model_predict(model_pipe, model_name, y_actual, 
                       y_pred ,y_pred_prob, path):
  """ save the model and training results"""
  if (model_pipe != None):
     # save the pipeline model and train results
    data_io.save_data(model_pipe, path, model_name+'_pipe_model_trained')

  # saving the validation results appending customer info and actual results
  validation_results = pd.DataFrame({ 
                      'highbrow_wines_actual': y_actual,
                      'highbrow_wines_prediction': y_pred,
                      'prediction_probability': y_pred_prob.round(3)
                      })
  #validation_results = pd.concat([customerinfo, df1], axis=1, join='inner')
    
  data_io.save_data(validation_results, path, model_name+'_model_results_train')
  data_io.save_data_csv(validation_results, path, model_name+'_model_results_train')

def recompute_predict_threshold(model_name, niteration, nthreshold, path):
  """ In case the probability threshold need to be changed and
      new predictions are recomputed, this function is used
      and results as saved as iterations"""
  # load the already predicted details
  pred_results = data_io.load_saved_data(path, model_name+'_model_results_train')
  y_actual = pred_results['highbrow_wines_actual']
  y_pred_initial = pred_results['highbrow_wines_prediction']
  y_prob_initial = pred_results['prediction_probability']
  customerid = pred_results.index.values
  
  # set the probability threshold
  y_predict_new = np.where(y_prob_initial > nthreshold, 1, 0)
  
  # evaluation step
  model_evaluation(y_actual, y_predict_new, y_prob_initial)

  # saving the model step
  save_model_predict(None, model_name+'_'+str(niteration), 
                     y_actual, y_predict_new ,y_prob_initial, path)
        
# plotting
def plot_confusion_matrix(y_true, y_pred,classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix (normalized)'
        else:
            title = 'Confusion matrix (without normalization)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return plt

# plotting
def plot_roc_curve(fpr, tpr, auc_score):
  """
    ROC curve plotting with AUC score
  """
  plt.plot(fpr, tpr,'r-',label = 'AUC score %.3f'%auc_score)
  plt.plot([0,1],[0,1],'k-',label='random')
  plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
  plt.legend()
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  return plt

def plot_prec_recall_curve(y_true, probas_pred):
  """plotting the precision and recall curve"""
  precision, recall, _ = precision_recall_curve(y_true, probas_pred)
  plt.step(recall, precision, color='b', alpha=0.2,where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
  return plt

def plot_featureimportance(model_pipe, model_name, feat_names):
  """plotting feature importance details after training the model"""
    
  model = model_pipe.named_steps[model_name]
  #model = model_pipe.steps[0][1]
  
  imp_vals = model.feature_importances_
  importance = {k:v for k,v in zip(feat_names, imp_vals)}
  importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

  sns.barplot(y=list(zip(*importance))[0], x=list(zip(*importance))[1])
  
  
def create_feature_map(features):
  outfile = open('xgb.fmap', 'w')
  i = 0
  for feat in features:
    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    i = i + 1
    
  outfile.close()

  
  #sorted_idx = np.argsort(model.feature_importances_)[::-1]
  #for index in sorted_idx:
  #  print([X_train.columns[index], model.feature_importances_[index]])
  #
  #plot_importance(model, max_num_features = 15)
  #plt.show()