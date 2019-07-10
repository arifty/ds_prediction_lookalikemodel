from sklearn.pipeline import Pipeline
from vecstack import stacking, StackingTransformer
from datetime import datetime, timedelta, date
import pandas as pd

import mod_03_featureselection as feat_select
import fn_data_io as data_io


# Classification model - Score
#----------------------------------------------------------------------------------------
def model_scoring_step(data_toscore, path, model_name='xgboost'):
  """Calling all other steps for data prep and model scoring
  """
  
  # logging
  print("Model scoring started_Timestamp:{}".format(format(str(datetime.now()))))
  
  # splitting features and train
  customerid = data_toscore['masked_customer_id']
  features_data = data_toscore.set_index('masked_customer_id')  
  labels = features_data['bought_highbrow_wines'].astype(int)
  features = features_data[features_data.columns.difference(['bought_highbrow_wines'])]
  
  # encode the categorical features and scale
  features_scaled = feat_select.data_preprocessing(features)
  multicoll_check = False
  
  # choose only the selected features before scoring the data
  (featureslist, X_scoring, 
   y_scoring) = feat_select.feature_selection_step(features_scaled,labels, 
                                                path, 'score', multicoll_check)
  
  
  # load the trained model saved in the directory
  model = data_io.load_saved_data(path, model_name+'_pipe_model_trained')
  
  # scoring with real data
  y_scoring_pred = model.predict(X_scoring)
  y_scoring_pred_prob = model.predict_proba(X_scoring)[:, 1]
  
  customerinfo = features.index.values
  
  # saving the model step
  save_model_predict(customerinfo, model, model_name, y_scoring_pred ,y_scoring_pred_prob, 
                     path)
  
  # logging
  print("Model scoring completed_Timestamp:{}".format(format(str(datetime.now()))))  
  
  
def save_model_predict(customerinfo, model, model_name, y_pred ,y_pred_prob, path):
  """ save the model & scoring results"""
    
  # saving the validation results appending customer info and actual results
  validation_results = pd.DataFrame({
                            'masked_customer_id': customerinfo,
                            'highbrow_wines_prediction': y_pred,
                            'prediction_probability': y_pred_prob.round(3)})
  validation_results = validation_results.set_index('masked_customer_id')
  #validation_results = pd.concat((customerinfo, df1), axis=1, join='inner')
    
  # save the pipeline model and train results
  data_io.save_data(model, path, model_name+'_pipe_model_scored')
  data_io.save_data(validation_results, path, model_name+'_model_results_scoring')
  data_io.save_data_csv(validation_results, path, model_name+'_model_results_scoring')
   
