  # fetch the feature importance
  rf_feature = rf.feature_importances(X_train,y_train)
  ada_feature = ada.feature_importances(X_train, y_train)
  xgb_feature = xgb.feature_importances(X_train,y_train)
  
  
  cols = X_train.columns.values
  # Create a dataframe with features
  feature_dataframe = pd.DataFrame( {'features': cols,
        'Random Forest feature importances': rf_features,
        'AdaBoost feature importances': ada_features,
        'Gradient Boost feature importances': xgb_features
      })
  

  # Create the new column containing the average of values
  feature_dataframe['mean'] = feature_dataframe.mean(axis= 1)
  feature_dataframe.head(3)
  
  # Plot the average of all feature_importance
  y = feature_dataframe['mean'].values
  x = feature_dataframe['features'].values
  data = [go.Bar(
              x= x,
               y= y,
              width = 0.5,
              marker=dict(
                 color = feature_dataframe['mean'].values,
              colorscale='Portland',
              showscale=True,
              reversescale = False
              ),
              opacity=0.6
          )]

  layout= go.Layout(
      autosize= True,
      title= 'Barplots of Mean Feature Importance',
      hovermode= 'closest',
  #     xaxis= dict(
  #         title= 'Pop',
  #         ticklen= 5,
  #         zeroline= False,
  #         gridwidth= 2,
  #     ),
      yaxis=dict(
          title= 'Feature Importance',
          ticklen= 5,
          gridwidth= 2
      ),
      showlegend= False
  )
  fig = go.Figure(data=data, layout=layout)
  py.iplot(fig, filename='bar-direct-labels')
  
