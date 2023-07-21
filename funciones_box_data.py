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
                              # scoring=('r2', 'neg_mean_squared_error'),
                              # return_train_score=True)

  print(f'Fit time mean: {cv_results["fit_time"].mean()}')
  print(f'Score time mean: {cv_results["score_time"].mean()}')
  print(f'Test score: {cv_results["test_score"]}')
  print(f'Test mean score: {cv_results["test_score"].mean()}')
  # print(cv_results['test_neg_mean_squared_error'])
  # print(cv_results['train_r2'])


def fit_model(model, x_train, y_train, x_test):
  try:
    x = x_train.values
    y = y_train.values
    x_test = x_test.values
  except AttributeError:
    x = x_train
    y = y_train.values
    x_test = x_test

  model.fit(x, y)
  ytest_pred = model.predict(x_test)
  print('Modelo entrenado en datos de entrenamiento')
  print('Return prediction')
  return ytest_pred


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


def one_hot_encode_with_custom_order(df):
  lista = [
      ['c_f', '/content/lista_acomodo_regiones.txt', ''],
      ['region_b1', '/content/lista_acomodo_regiones.txt', '.1'],
      ['region_b2', '/content/lista_acomodo_regiones.txt', '.2'],
      ['boxstyle_b1', '/content/lista_estilos_pelea.txt', ''],
      ['boxstyle_b2', '/content/lista_estilos_pelea.txt', '.1'],
      ['stance_b1', '/content/lista_postura.txt', ''],
      ['stance_b2', '/content/lista_postura.txt', '.1']
  ]
  df_1 = df.copy()

  for i in lista:
    col = i[0]
    order = i[1]
    suf = i[2]

    with open(f'fitted_vectorizer_{col}_.pkl', 'rb') as file:
      vectorizer = pkl.load(file)

    one_hot_vectors = vectorizer.transform(df[col])
    feature_names = vectorizer.get_feature_names_out()

    with open(order, 'r') as file:
        custom_order = file.read()

    custom_order = [i for i in custom_order.split(',')][:-1]
    if suf == '':
      print(f'{col} : sin sufijo')
    else:
      custom_order = [i+suf for i in custom_order]
      print(f'{col} : sufijo : {suf}')

    temp = pd.DataFrame(one_hot_vectors.toarray(), columns=custom_order)
    ordered_temp = temp[custom_order]

    df_1 = pd.concat([df_1, ordered_temp], axis=1)
    df_1.drop(columns=col, inplace=True)

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
    df_pred['pelea'] = pelea.values
    df_pred['boxer1_pred'] = [i.astype(str).replace('-1','loss').replace('1','win') for i in list(pred_new)]
    df_pred['prob_win'] = probabilities[:,1]
    df_pred['prob_loss'] = probabilities[:,0]
    return df_pred
