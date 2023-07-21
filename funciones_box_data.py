import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_validate

def cross_val(model, x_train, x_val, y_train, y_val, cv=5):
  try:
    X = pd.concat([x_train, x_val])
    y = pd.concat([y_train, y_val])
  except TypeError:
    X = pd.concat([pd.DataFrame(x_train), pd.DataFrame(x_val)])
    y = pd.concat([y_train, y_val])

  cv_results = cross_validate(model, X.values, y.values, cv=cv)

  print(f'Fit time mean: {cv_results["fit_time"].mean()}')
  print(f'Score time mean: {cv_results["score_time"].mean()}')
  print(f'Test score: {cv_results["test_score"]}')
  print(f'Test mean score: {cv_results["test_score"].mean()}')


import pickle as pkl

def feat_eng(df):
  df = df.copy()
  df['b1_momios_menores'] = np.where(df.b1_bet < df.b2_bet, 1, 0)
  df['b1_mas_wins'] = np.where(df.b1_w > df.b2_w, 1, 0)
  df['b1_menos_draws'] = np.where(df.b1_d < df.b2_d, 1, 0)
  df['b1_menos_loss'] = np.where(df.b1_l < df.b2_l, 1, 0)
  df['b1_mas_ko_perc'] = np.where((df.b1_wk / df.b1_w) > (df.b2_wk / df.b2_w), 1, 0)
  df['b1_invicto'] = np.where(df.b1_l <= 0, 1, 0)
  df['b2_invicto'] = np.where(df.b2_l <= 0, -1, 0)
  df['b1_mas_3_loss'] = np.where(df.b1_l > 3, -1, 0)
  df['b2_mas_3_loss'] = np.where(df.b2_l > 3, 1, 0)

  return df


def one_hot_encoder(df, encoder1, encoder2, encoder3):
  df_1 = df.copy()
  columns = ['c_f', 'region_b1', 'region_b2']
  for i in columns:
    one_hot_vectors = encoder1.transform(df[i])
    feature_names = encoder1.get_feature_names_out()
    temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
    df_1 = pd.concat([df_1,temp], axis=1)
    df_1.drop(columns=i, inplace=True)

  columns = ['stance_b1', 'stance_b2']
  for i in columns:
    one_hot_vectors = encoder3.transform(df[i])
    feature_names = encoder3.get_feature_names_out()
    temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
    df_1 = pd.concat([df_1,temp], axis=1)
    df_1.drop(columns=i, inplace=True)

  columns = ['boxstyle_b1', 'boxstyle_b2']
  for i in columns:
    one_hot_vectors = encoder2.transform(df[i])
    feature_names = encoder2.get_feature_names_out()
    temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
    df_1 = pd.concat([df_1,temp], axis=1)
    df_1.drop(columns=i, inplace=True)

  return df_1


from sklearn.base import BaseEstimator, TransformerMixin

class Data_clusterer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clusters = self.model.predict(X)
        clusters = pd.Series(clusters, index=X.index, name='cluster')
        temp = X.merge(clusters, how='outer', left_index=True, right_index=True)
        return temp


class Model_applied(BaseEstimator, TransformerMixin):
  def __init__(self, model):
    self.model = model

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    df_pred = pd.DataFrame()
    pred_new = model.predict(X)
    probabilities = model.predict_proba(X)
    df_pred['model_pred'] = pred_new
    df_pred['boxer1_pred'] = [i.astype(str).replace('-1','loss').replace('1','win') for i in list(pred_new)]
    df_pred['prob_win'] = probabilities[:,1]
    df_pred['prob_loss'] = probabilities[:,0]
    return df_pred


def check_fails_and_probas(df, y_true, y_pred, prob_loss, prob_win, figsize=(5,3)):
  dfx = xl_test_clus.copy()
  dfx['true_res'] = yl_test.result.values
  dfx['pred_res'] = yl_test_pred
  dfx['goodpred'] = (dfx.true_res == dfx.pred_res)
  dfx['prob_loss'] = probabilities[:,0]
  dfx['prob_win'] = probabilities[:,1]

  # Count the occurrences of each cluster and goodpred combination
  counts = dfx.groupby(['cluster', 'goodpred']).size().reset_index(name='count')

  plt.figure(figsize=figsize)
  plt.bar(counts.index, counts['count'])
  labels = [f'Cluster {c}, GoodPred {g}' for c, g in zip(counts['cluster'], counts['goodpred'])]
  plt.xticks(counts.index, labels, rotation=90)
  plt.xlabel('Cluster and GoodPred')
  plt.ylabel('Count')
  plt.title('Data Counts by Cluster and GoodPred')
  plt.show()

  # perc false per true by clusters
  titulo = 'Porcentaje de False per Trues by clusters'
  print(titulo+'\n'+(len(titulo)*'='))
  perc_false_per_true_by_cluster = counts.groupby('cluster').apply(lambda x: x[x['goodpred'] == False]['count'].sum() / x[x['goodpred'] == True]['count'].sum())
  print(perc_false_per_true_by_cluster)


  fig,ax=plt.subplots(figsize=figsize)
  sns.histplot(dfx[dfx.goodpred==True].prob_win)
  ax.set_title('Cantidad de Trues según win pred value')
  plt.show()

  fig,ax=plt.subplots(figsize=(5,3))
  sns.histplot(dfx[dfx.goodpred==False].prob_win)
  ax.set_title('Cantidad de False según win pred value')
  plt.show()
