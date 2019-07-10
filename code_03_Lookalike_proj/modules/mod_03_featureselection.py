# Feature selection #

# import packages
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,RobustScaler
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# import custom usage modules
import fn_featselect_weightofevid as woe
import fn_data_io as data_io


def feature_selection_step(features, labels, path, exec_type='train', multicoll_check=False):
  """perform voting based selection for best features using below selection
     1: feature selection with Information Value using Weight of evidence
     2: feature selection with Random Forest feature importance
     3: feature selection with Chi-square best score
     4: feature selection with RFE of Logit algo
     5: combine and vote from above 4 outputs
     6: remove multi-collinear features from final list
  """

  feature_list = list(features.columns)
  scaled_combined_data = pd.concat([features, labels], axis=1, join='inner')

  # feature selection procedure only incase of model training
  if (exec_type == 'train'):
    # call function to perform feature selection procedure
    (features_final, selected_features) = select_best_features(features, 
                                        labels, scaled_combined_data, multicoll_check)
    
    #features_final = features
    featureslist_selected = list(features_final.columns)
    
    #featureslist_selected = ['cat_KoudeSauzen',
    #                'cat_Babyluiers',
    #                'total_revenue',
    #                'cat_Ontbijtgranen_Volwassenen',
    #                'SOW_colr',
    #                'cat_VerseKaasFruitkazen',
    #                'SOW_type_colr_UNKNOWN',
    #                'SOW_type_colr_SOW90-100',
    #                'cat_VisGerookt',
    #                'cat_KaasSeizoenskazen',
    #                'cat_AP_STDR_WhiskyONLINE',
    #                'cat_Kauwgum',
    #                'cat_AP_STDR_PortoONLINE',
    #                'total_discount_format',
    #                'cat_VNCBerBurgers',
    #                'cat_VNCWildSteak',
    #                'cat_MelkKarnemelk',
    #                'SOW_type_colr_SOW_100+',
    #                'n_cogo',
    #                'cat_bbqfoodevent',
    #                'cat_Chips',
    #                'price_sens_colr_format',
    #                'cat_ParfumerieEHBO',
    #                'cat_Notengedroogdfruit_groenten',
    #                'cogo_rev',
    #                'Collishop_customer_Y',
    #                'HOUSEHOLDTYPOLOGY_g_HHnochild_55_plus',
    #                'HOUSEHOLDTYPOLOGY_c_Single_55_plus',
    #                'cat_Wijn_Stillewijnen_RAYON'
    #          ]
    
    
    # save the seleted feature list, also to be used for scoring the model
    data_io.save_data(featureslist_selected, path, 'featureslist_selected')
    data_io.save_data_csv(selected_features, path, 'selected_features_voting')
  
  # loading the feature list for both training and scoring
  featureslist_selected = data_io.load_saved_data(path, 'featureslist_selected')
      
  # save the selected feature data
  data_io.save_data(features[featureslist_selected], path, 'features_'+exec_type)  
  features_final = data_io.load_saved_data(path, 'features_'+exec_type)

  data_io.save_data(labels, path, 'labels_'+exec_type)
  labels = data_io.load_saved_data(path, 'labels_'+exec_type)
  
  return (featureslist_selected, features_final, labels)
  #return (feature_list, features, labels)
  

def data_preprocessing(data_toprocess):  
  """ DATA PRE-PROCESSING STEP (encoding and scaling)"""
  # get the categorical features to do OneHotEncoding
  cat_features = data_toprocess.select_dtypes(include=['object']).columns.tolist()
  
  # define the preprocessor for OneHotEncoding
  encoded_data = pd.get_dummies(data_toprocess,columns=cat_features,drop_first=True)
    
  # scaling the data (using MinMaxScaler to get positive values)
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(encoded_data)
  scaled_data = pd.DataFrame(scaled_data, index=encoded_data.index, columns=list(encoded_data.columns))
  
  show_scaled_plots(data_toprocess, scaled_data, 
                  ['cogo_rev','n_cogo','total_discount_format','total_revenue'])
  
  return scaled_data

def show_scaled_plots(feat_bef_scale, feat_aft_scale, feat_cols):
  # visualization
  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 7))
  
  ax1.set_title('Before Scaling')
  ax2.set_title('After Scaling')
  
  for feat in feat_cols:
    sns.kdeplot(feat_bef_scale[feat], ax=ax1)
    sns.kdeplot(feat_aft_scale[feat], ax=ax2)

  plt.show()

  
def featselect_woe(combined_data, labels):
  """Get Information variable according to weight of evidence"""
  final_iv, IV = woe.data_vars(combined_data,labels)
  IV = IV.rename(columns={'VAR_NAME':'index'})

  IV.sort_values(['IV'],ascending=0)
  
  return IV


def featselect_rfi(feat, labels):
  """Get Feature Importance using Random Forest"""
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  
  rf_clf = RandomForestClassifier()
  rf_clf.fit(feat,labels)

  preds = rf_clf.predict(feat)
  
  # check the accuracy score
  accuracy = accuracy_score(preds,labels)
  print(accuracy)

  rfi = pd.DataFrame(rf_clf.feature_importances_, columns = ["RF"], index=feat.columns)
  rfi = rfi.reset_index()

  return rfi.sort_values(['RF'],ascending=0)

def featselect_chi2(feat, labels):
  """Get Chi Square score using SelectKBest algo"""
  from sklearn.feature_selection import SelectKBest, chi2

  model = SelectKBest(score_func=chi2, k=5)
  fit = model.fit(feat, labels)
  
  np.set_printoptions(suppress=True)
  print(fit.scores_)

  pd.options.display.float_format = '{:.2f}'.format
  chi_sq = pd.DataFrame(fit.scores_, columns = ["Chi_Square"], index=feat.columns)
  chi_sq = chi_sq.reset_index()
  
  return chi_sq.sort_values('Chi_Square',ascending=0)
  
def featselect_l1(feat, labels):
  """Get L1 penalty score using Linear SVM algo"""
  from sklearn.svm import LinearSVC
  from sklearn.feature_selection import SelectFromModel

  lsvc = LinearSVC(C=1.0, penalty="l1", dual=False, tol=1e-5, max_iter=4000).fit(feat, labels)
  model = SelectFromModel(lsvc,prefit=True)

  l1 = pd.DataFrame(model.get_support(), columns = ["L1"], index=feat.columns)

  l1 = l1.reset_index()
  l1[l1['L1'] == True]
  
  return l1

def featselect_recurFE(feat, labels):
  """Get Recursive Feature Selection using LogisticRegression algo"""
  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LogisticRegression

  model = LogisticRegression()
  rfe = RFE(model, 20)
  fit = rfe.fit(feat, labels)
  
  selected_feat = pd.DataFrame(rfe.support_, columns = ["FE"], index=feat.columns)
  selected_feat = selected_feat.reset_index()
  
  selected_feat[selected_feat['FE'] == True]
  return selected_feat


def check_multicol_feat(feat, vif_threshold=10):
  """Check the Multicollinearity between the selected features- VIF"""
  vif = woe.calculate_vif(feat)
  while vif['VIF'][vif['VIF'] > vif_threshold].any():
      remove = vif.sort_values('VIF',ascending=0)['Features'][:1]
      final_feat = feat.drop(remove,axis=1)
      vif = woe.calculate_vif(final_feat)

  return final_feat
  
def select_best_features(feat, labels, combined_feat, multicoll_check):
  """returns the final feature dataset and the ranking of selected features"""
  from functools import reduce
  
  # call all feature selection functions
  iv = featselect_woe(combined_feat, labels)
  rfi = featselect_rfi(feat, labels)
  chi_sq = featselect_chi2(feat, labels)
  #l1 = featselect_l1(feat, labels) #getting convergence error
  fe = featselect_recurFE(feat, labels)
  
  # combine all the feature selection scores
  dfs = [iv, rfi, chi_sq, fe]
  all_scores = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs)
  all_scores.head(10)
  
  # VARIABLE SCORING AND VOTING
  columns = ['IV', 'RF', 'Chi_Square']

  score_table = pd.DataFrame({},[])
  score_table['index'] = all_scores['index']

  for i in columns:
      score_table[i] = all_scores['index'].isin(list(all_scores.nlargest(5,i)['index'])).astype(int)

  score_table['FE'] = all_scores['FE'].astype(int)


  score_table['final_score'] = score_table.sum(axis=1)
  score_table.sort_values('final_score',ascending=0)

  # select the features whose final_score > 0
  selected_features = score_table[score_table['final_score']>0]
  sel_featurelist = selected_features['index'].tolist()
  
  # getting the final features dataset
  features_bef_collin = feat[sel_featurelist]
  if (multicoll_check): feat_final = check_multicol_feat(features_bef_collin)
  else: feat_final = features_bef_collin
  
  return (feat_final, selected_features)

  