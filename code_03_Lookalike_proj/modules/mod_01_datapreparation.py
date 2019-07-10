import pandas as pd

def remove_target_null(input_data, target_var):
  """exclude the unlabelled data (null values in response variable)"""
  input_data = input_data[~input_data[target_var].isnull()]
  return input_data

def fixing_unwanted_data(input_data):
  """drop 'jaar' column and transform HOUSEHOLDTYPOLOGY"""
  input_data = input_data.drop('jaar', axis=1)  
  input_data.loc[input_data['HOUSEHOLDTYPOLOGY']=='!','HOUSEHOLDTYPOLOGY'] = 'UNKNOWN'
  input_data.loc[input_data['SOW_type_colr']=='!','SOW_type_colr'] = 'UNKNOWN'
  
  return input_data

def enrich_missing_values(input_data):  
  """enrich the missing values for 'Collishop_customer' and 'total_revenue'"""
  input_data['Collishop_customer'] = input_data['Collishop_customer'].fillna('N')

  input_data['total_revenue'] = pd.to_numeric(input_data['total_revenue'], errors='coerce')
  input_data['total_revenue'] = input_data['total_revenue'] \
                                      .fillna(value=input_data['total_revenue'].mean())
    
  input_data.update(input_data['prod_ticket'] \
                    .fillna(value=input_data['prod_ticket'].mean().round()))
  
  return input_data

def clip_negative_values(input_data):  
  """clip the all revenue turnover columns to zero incase of negative"""
  input_data.update(input_data.filter(regex="^cat_.*$").clip(lower=0))
  input_data.update(input_data[['total_revenue','rev_ticket','cogo_rev']].clip(lower=0))  
  return input_data

def replacenulls_catcols(input_data):
  input_data.update(input_data.filter(regex="^cat_.*$").fillna(0))
  return input_data

def convert_negative_positive(input_data, col_name):
  """price sensitivity and discount columns transform, convert negative to positive and scale.
     create a new column and drop original col to have traceability"""
  input_data[col_name+'_format'] = input_data[col_name]+abs(input_data[col_name].min())
  input_data.drop(col_name, axis=1, inplace=True)    
  return input_data
  
def remove_SOW_outliers(input_data):
  """'SOW_type_colr' has 2 outliers for frequency and revenue.
     removing those observations as we already know they are outliers"""
  input_data = input_data[input_data['SOW_type_colr'] != 'Outlier_om']
  input_data = input_data[input_data['SOW_type_colr'] != 'Outlier_fr']  
  return input_data


def transform_step(input_data, target_var):
  """ tranforms the input_data with below pre-defined cleansing steps 
      1: remove target variable null
      2: fixing_unwanted_data
      3: enriching missing values
      4: replace nulls for category columns
      5: transform negative values
      6: converting and scaling price sensitive column
      7: removing outliers
  """
  transform_data = remove_target_null(input_data, target_var)
  transform_data = fixing_unwanted_data(transform_data)
  transform_data = enrich_missing_values(transform_data)
  transform_date = replacenulls_catcols(transform_data)
  transform_data = clip_negative_values(transform_data)
  transform_data = convert_negative_positive(transform_data, 'price_sens_colr')
  transform_data = convert_negative_positive(transform_data, 'total_discount')
  transform_data = remove_SOW_outliers(transform_data)
  
  return transform_data
