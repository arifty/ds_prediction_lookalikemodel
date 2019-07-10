import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def check_target_distribution(input_data):
  target_var_dist = input_data['bought_highbrow_wines'].value_counts()
  
  print('target variable distribution:')
  print('class '+str(target_var_dist.index[0])+ ' : ' +str(target_var_dist[0]))
  print('class '+str(target_var_dist.index[1])+ ' : ' +str(target_var_dist[1]))  
  print('Class distribution ratio:',round(target_var_dist[0]/target_var_dist[1], 2),': 1')
  target_var_dist.plot(kind='bar', title='Class distribution')

  
def check_outliers(input_data):

  freq_outlier = input_data[input_data['SOW_type_colr']=='Outlier_fr'] \
          ['masked_customer_id'].count()
  
  turnover_outlier = input_data[input_data['SOW_type_colr']=='Outlier_om'] \
          ['masked_customer_id'].count()
  
  print('Outlier counts:')
  print('Frequency outlier: '+str(freq_outlier))
  print('Turnover_outlier: '+str(turnover_outlier))


def plot_counts(input_data, x_variable):
  ax = sns.countplot(x=x_variable,data=input_data,palette='viridis')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  plt.tight_layout()


def plot_facetgrid(input_data):
  facet_data = sns.FacetGrid(data=input_data,col='bought_highbrow_wines')
  facet_data.map(sns.distplot,'total_revenue',kde=False)

def plot_pairgrid(input_data):
  pairgrid_data = sns.PairGrid(input_data)
  pairgrid_data.map_diag(sns.distplot)
  pairgrid_data.map_upper(plt.scatter)
  pairgrid_data.map_lower(sns.kdeplot)

  
def plot_corr(input_data):
  cor = data_class1.corr()
  plt.figure(figsize=(12,6))
  sns.heatmap(cor,cmap='Set1',annot=True)


def exploration_step(input_data):
  
  # check the stats with target distribution
  check_target_distribution(input_data)
  check_outliers(input_data)
  
  # split the classes
  data_class0 = input_data[input_data['bought_highbrow_wines']==0.0]
  data_class1 = input_data[input_data['bought_highbrow_wines']==1.0]
  
  # perform categorical variable analysis
  sns.distplot(data_class0['total_revenue'],color='red',kde=False,bins=50)
  sns.distplot(data_class1['total_revenue'],color='blue',kde=False,bins=50)
  
  # histograms for turnover per category
  turnover_percat_cols = ['cat_AP_STDR_WhiskyONLINE','cat_Wijn_Stillewijnen_RAYON',
                                'cat_Chips','cat_bbqfoodevent']
  data_class1[turnover_percat_cols].hist(bins=30)
  
  # relation between numerical variables
  sns.jointplot(x=data_class1['total_revenue'], y=data_class1['cat_Wijn_Stillewijnen_RAYON'])
  sns.jointplot(x=data_class1['cogo_rev'], y=data_class1['cat_Wijn_Stillewijnen_RAYON'])

  # analyze categorical variables
  plot_counts(data_class0, 'HOUSEHOLDTYPOLOGY')
  plot_counts(data_class1, 'HOUSEHOLDTYPOLOGY')
  
  plot_counts(data_class0, 'SOW_type_colr')
  plot_counts(data_class1, 'SOW_type_colr')


















#
#
#plot_facetgrid(transformed_data)
#
#sns.heatmap(categ_rev_dist,cmap='coolwarm',linecolor='white',linewidth=2)
#
#
#
#summed = full_df.groupby(["Group", "Cluster", "Week"])["Slot Request"].sum().reset_index() #reset_index turns this back into a normal dataframe
#g = sns.FacetGrid(summed, col="Group") #create a new grid for each "Group"
#g.map(sns.pointplot, 'Week', 'Slot Request') #map a pointplot to each group where X is Week and Y is slot request
#
#
#sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',split=True)
#
#
#sns.pairplot(transformed_data[['cat_AP_STDR_WhiskyONLINE','cat_Wijn_Stillewijnen_RAYON',
#                                'cat_Chips','cat_bbqfoodevent']])
#plt.show()
#
#cat_AP_STDR_WhiskyONLINE
#cat_Wijn_Stillewijnen_RAYON
#cat_Chips
#cat_bbqfoodevent
#
#
#household_dist = transformed_data.groupby(['bought_highbrow_wines','HOUSEHOLDTYPOLOGY']).count()
#household_dist = household_dist.reset_index()
#
#
#sns.barplot(x="HOUSEHOLDTYPOLOGY", y="total_revenue", hue="bought_highbrow_wines",
#            data=household_dist)
#
#
#
#
#sns.lmplot(x='HOUSEHOLDTYPOLOGY',y='total_revenue',hue='bought_highbrow_wines',data=household_dist)
#
#
#
#%matplotlib inline
#household_dist = sns.FacetGrid(data=transformed_data,col='bought_highbrow_wines')
#household_dist.map(sns.distplot,'HOUSEHOLDTYPOLOGY',kde=False)
#plt.show()
#
#  
#
#  
#  
#
#
#raw_input_data.describe()
#  
## validation  
#transformed_data[transformed_data['SOW_type_colr']=='!'][['SOW_type_colr','SOW_colr']].describe()
#
#
#  
#  
#  
#  
## set the customer id as index
##labelled_data.set_index('masked_customer_id', inplace=True)
#
## validation
#
#labelled_data['Collishop_customer'].value_counts()
#
#labelled_data[labelled_data['SOW_type_colr']=='Outlier_fr'] \
#          [['masked_customer_id','bought_highbrow_wines','SOW_type_colr','SOW_colr']].count()
#
#labelled_data[labelled_data['masked_customer_id']==348924] \
#          [['masked_customer_id','bought_highbrow_wines','SOW_type_colr','SOW_colr']].count()
#  
#
#labelled_data['masked_customer_id'].nunique()
#labelled_data['masked_customer_id'].count()
#  
#labelled_data.head(6)
#
## check the missing values
#labelled_data.isnull().mean().sort_values(ascending=False)*100
#
## check the correlation between variables
#corr = labelled_data.corr()
#sns.heatmap(corr, 
#        xticklabels=corr.columns,
#        yticklabels=corr.columns)
#
#
#
## formatting
#
#target_data = labelled_data.drop('jaar', axis=1)
#target_data.head(6)
#
#target_data.loc[target_data['HOUSEHOLDTYPOLOGY']=='!','HOUSEHOLDTYPOLOGY'] = 'UNKNOWN'
#target_data['HOUSEHOLDTYPOLOGY'].unique()
#
## check the null value columns
#null_columns=target_data.columns[target_data.isnull().any()]
#null_columns
#
## check the null values
#target_data[target_data['total_revenue'].isnull()][['total_revenue']]
#
#
## enrich the missing values
#target_data['Collishop_customer'] = target_data['Collishop_customer'].fillna('N')
#
#target_data['total_revenue'] = pd.to_numeric(target_data['total_revenue'], errors='coerce')
#target_data['total_revenue'] = target_data['total_revenue'] \
#                                    .fillna(value=target_data['total_revenue'].mean())
#
## check negative values
#target_data[target_data < 0]
#target_data.min()
#
#
## clip the category turnovers to zero incase of negative
#target_data.update(target_data.filter(regex="^cat_.*$").clip(lower=0))
#
#
## price sensitivity column transform
#target_data['price_sens_colr_format'] = target_data['price_sens_colr']+abs(target_data['price_sens_colr'].min())
#target_data.drop('price_sens_colr', axis=1, inplace=True)
#
## convert the negative discount values to positive
#target_data['total_discount'] = target_data['total_discount'].abs()
#
#target_data.update(target_data[['total_revenue','rev_ticket','cogo_rev']].clip(lower=0))
#
#
## removing the outliers
#
#target_data[target_data['SOW_type_colr'] == 'Outlier_om'].count()
#target_data[target_data['SOW_type_colr'] == 'Outlier_fr'].count()
#
#            
#target_data = target_data[target_data['SOW_type_colr'] != 'Outlier_om']
#target_data = target_data[target_data['SOW_type_colr'] != 'Outlier_fr']
#            
#
## ------- PAUSE -----------
## save the data
#pickle.dump(target_data, open(output_dir+ '/target_data.p', 'wb'))
#
#target_data = pickle.load(open(output_dir+ '/target_data.p', "rb" ))
## ------- PAUSE -----------
#  
#  
#
## check negative values
#contin_data = labelled_data[labelled_data.select_dtypes(include=['float', 'int', 'double']).columns.tolist()]
#
#contin_data.min()
#contin_data['total_revenue'].describe()
#
#contin_data[contin_data < 0].columns
#contin_data[contin_data < 0].isnull().sum().sum()
#
#contin_data[contin_data.min() < 0]
#
#contin_data[contin_data['total_discount'] >0][['total_discount']].count()
#
#contin_data.loc[contin_data['total_revenue'] <0, 'total_revenue'] = contin_data['total_revenue'].abs()
#contin_data.loc[contin_data['rev_ticket'] >0, 'rev_ticket'].sum()
#contin_data.loc[contin_data['cogo_rev'] >0, 'cogo_rev'].sum()
#
#contin_data.loc[contin_data['price_sens_colr'] <0, 'price_sens_colr'].count()
#
#contin_data['price_sens_colr'] = contin_data['price_sens_colr']+abs(contin_data['price_sens_colr'].min())
#
#
#
#contin_data['total_discount'].abs()
#
#
#contin_data.loc[contin_data<0] = contin_data
#
#labelled_data.select_dtypes(include=['float', 'int', 'double']).columns.tolist()
#labelled_data.info()
#
#  
#
#
#
#
#
## Some Data validation steps
## --------------------------
## Yes = 9959, No = 190037
#labelled_data[(labelled_data['bought_highbrow_wines']==0) &
#             (labelled_data['Collishop_customer']=='Y')].count()
#
#
