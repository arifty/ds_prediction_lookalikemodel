
import fn_data_io as dio
import mod_04_modeltraining as model_train
from xgboost import XGBClassifier
import seaborn as sns
import operator

features_train = dio.load_saved_data(output_dir, 'features_train')
labels_train = dio.load_saved_data(output_dir, 'labels_train')
customerid_train = dio.load_saved_data(output_dir, 'customerid_train')

model_pipe = dio.load_saved_data(output_dir, 'xgboost_pipe_model_trained')
kfold_validate(model, X_tr, y_tr, 10, 'roc_auc_score')

path = output_dir
model_name = 'stacking'
pred_results = dio.load_saved_data(path, model_name+'_model_results_train')
y_test = pred_results['highbrow_wines_actual']
y_pred = pred_results['highbrow_wines_prediction']
y_pred_prob = pred_results['prediction_probability']



# check the feature importance plot
features_train = dio.load_saved_data(output_dir, 'features_train')
features_list = features_train.columns.tolist()
model_name = 'xgboost'
model_pipe = dio.load_saved_data(output_dir+'/FINAL_Models/XGB/', model_name+'_pipe_model_trained')

model_train.plot_featureimportance(model_pipe, 'xgboost', features_list)




model = model_pipe.named_steps[model_name]
model_estim = model.estimators[2][1]

dio.save_data(model_estim, output_dir, 'stacking_pipe_xgb')
model_estim = dio.load_saved_data(output_dir, 'stacking_pipe_xgb')
model_pipe = model_estim.fit(features_train, labels_train)


#model = model_pipe.steps[0][1]
#model_estim.get_booster().get_fscore()
imp_vals = model.feature_importances_
importance = {k:v for k,v in zip(features_list, imp_vals)}
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)


sns.barplot(y=list(zip(*importance))[0], x=list(zip(*importance))[1])




importance = dict(sorted(importance.items()))

importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

sns.barplot(y=list(features_train.loc[:, featureslist_train].columns), 
            x=list(importance))

sns.barplot(y=importance.key), 
            x=list(importance))

from xgboost import plot_importance
plot_importance(model_estim.booster(), )

  plot_df = pd.DataFrame(imp_vals, columns=['feature', 'Importance'])
  
  plt.figure()
  plot_df.plot()
  plot_df.plot(kind='barh', x='feature', y='Importance', legend=False, figsize=(6, 10))
  plt.title('XGBoost Feature Importance')
  plt.xlabel('relative importance')
  plt.gcf().savefig(path+'/feature_importance_xgb.png')
  plt.show()



target_count = target_data['bought_highbrow_wines'].value_counts()
target_count
print('Class distribution ratio:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count of Class distribution')



# encode the categorical features
cat_features = target_data.select_dtypes(include=['object']).columns.tolist()
encoded_data = pd.get_dummies(target_data,columns=cat_features,drop_first=True)


features = encoded_data[encoded_data.columns.difference(['bought_highbrow_wines'])]
labels = encoded_data['bought_highbrow_wines']


plot_2d_space(features, labels, 'Imbalanced dataset (2 PCA components)')



from sklearn.decomposition import PCA

def run_pca (X, nr_comonents =2):
  pca = PCA(n_components=nr_comonents)
  X = pca.fit_transform(X)
  
  return X


plot_2d_space(run_pca(features), labels, 'Imbalanced dataset (2 PCA components)')



#Over-sampling: SMOTE

from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X_train, y_train)

X_train_res = pd.DataFrame(X_sm, columns=X_train.columns.tolist())
y_train_res = pd.Series(y_sm)

y_train_res.value_counts()


X_pca = run_pca(X_sm)
y = y_sm

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')



def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
null_columns=features_score.columns[features_score.isnull().any()]
null_columns
features_score[null_columns].isnull().sum()