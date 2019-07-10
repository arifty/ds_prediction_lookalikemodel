    
#  # Put in our parameters for classifiers
#  # Random Forest parameters
#  rf_params = {
#      'n_jobs': -1,
#      'n_estimators': 100,
#      'warm_start': True, 
#      'max_features': 'auto',
#      'max_depth': 10,
#      'min_samples_split': .3,
#      'verbose': 0
#  }
#
#
#  # AdaBoost parameters
#  ada_params = {
#      'n_estimators': 100,
#      'learning_rate' : 0.75
#  }
#
#  # Gradient Boosting parameters
#  xgb_params = {
#      'n_estimators': 100,
#      'colsample_bytree':0.8,
#      'gamma':0.5,
#      'learning_rate':0.7,
#      'max_depth':5,
#      'min_child_weight':1.5,
#      'reg_alpha':0.75,
#      'reg_lambda':0.45,
#      'nthread':6, 
#      'scale_pos_weight':1,
#      'subsample':0.9
#  }
#  
#
#  # Support Vector Classifier parameters 
#  svc_params = {
#      'kernel' : 'linear',
#      'gamma': 'scale',
#      'C' : 0.025
#      }
#  

#  # Extra Trees Parameters
#  et_params = {
#      'n_jobs': -1,
#      'n_estimators':100,
#      #'max_features': 0.5,
#      'max_depth': 10,
#      'min_samples_leaf': 2,
#      'verbose': 0
#  }
  
#  # Create 5 objects that represent our 4 models
#  rf = sklearn_helper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
#  ada = sklearn_helper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
#  xgb = sklearn_helper(clf=XGBClassifier, seed=SEED, params=xgb_params)
#  svc = sklearn_helper(clf=SVC, seed=SEED, params=svc_params)
#  
#  
#  models = [rf, ada, xgb, svc]
#  
#  # stack the models
#  S_train, S_test = stacking(models,
#                             X_train, y_train, X_test,
#                             regression=False,
#                             mode='oof_pred_bag',
#                             needs_proba=False,
#                             save_dir=None,
#                             metric=accuracy_score,
#                             n_folds=10,
#                             stratified=True,
#                             shuffle=True,
#                             random_state=0,
#                             verbose=2)
#  
#  # ------- PAUSE -----------
#  # save the data
#  pickle.dump(S_train, open(output_dir+ '/Stacked_train.p', 'wb'))
#  pickle.dump(S_test, open(output_dir+ '/Stacked_test.p', 'wb'))
#
#  S_train = pickle.load(open(output_dir+ '/Stacked_train.p', "rb" ))
#  S_test = pickle.load(open(output_dir+ '/Stacked_test.p', "rb" ))
#  # ------- PAUSE -----------


#feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
#feat_imp.plot(kind='bar', title='Feature Importances')
#plt.ylabel('Feature Importance Score')

#def importance_XGB(clf):
#impdf = []
#for ft, score in clf.booster().get_fscore().iteritems():
#impdf.append({‘feature’: ft, ‘importance’: score})
#impdf = pd.DataFrame(impdf)
#impdf = impdf.sort_values(by=’importance’, ascending=False).reset_index(drop=True)
#impdf[‘importance’] /= impdf[‘importance’].sum()
#return impdf
#
#importance_XGB(xgb1)
  
