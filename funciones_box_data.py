import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_validate

def cross_val(model, x_1, x_2, y_1, y_2, cv=5):
  '''
  Función para hacer cross validate de modelos ml.
  :model: modelo a revisar.
  :x_1: data x 1 (entrenamiento).
  :x_2: data x 2 (testeo).
  :y_1: data y 1 (entrenamiento).
  :y_2: data y 2 (testeo).
  :cv: cross val folders, default=5.
  :print: métricas de resultados.
  '''
  try:
    X = pd.concat([x_1, x_2])
    y = pd.concat([y_1, y_2])
  except TypeError:
    X = pd.concat([pd.DataFrame(x_1), pd.DataFrame(x_2)])
    y = pd.concat([y_1, y_2])

  cv_results = cross_validate(model, X.values, y.values, cv=cv)

  print(f'Fit time mean: {cv_results["fit_time"].mean()}')
  print(f'Score time mean: {cv_results["score_time"].mean()}')
  print(f'Test score: {cv_results["test_score"]}')
  print(f'Test mean score: {cv_results["test_score"].mean()}')


import pickle as pkl

def feat_eng(df):
  '''
  Feature engineering values.
  :df: Dataframe sobre el cual agregar las características.
  :return: df con las características.
  '''
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


from sklearn.base import BaseEstimator, TransformerMixin


class One_hot_encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder1, encoder2, encoder3):
      '''
      Vectorizador de string features.
      :encoder1: vectorizador regiones.
      :encoder2: vectorizador posturas.
      :encoder3: vectorizador estilos.
      '''
      self.encoder1 = encoder1
      self.encoder2 = encoder2
      self.encoder3 = encoder3

    def fit(self, X, y=None):
      return self

    def transform(self, X):
      '''
      :X: dataframe base.
      :return: df con las columnas vectorizadas.
      '''
      df_1 = X.copy()
      columns = ['c_f', 'region_b1', 'region_b2']
      for i in columns:
        one_hot_vectors = self.encoder1.transform(df_1[i])
        feature_names = self.encoder1.get_feature_names_out()
        temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
        df_1 = pd.concat([df_1,temp], axis=1)
        df_1.drop(columns=i, inplace=True)

      columns = ['stance_b1', 'stance_b2']
      for i in columns:
        one_hot_vectors = self.encoder2.transform(df_1[i])
        feature_names = self.encoder2.get_feature_names_out()
        temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
        df_1 = pd.concat([df_1,temp], axis=1)
        df_1.drop(columns=i, inplace=True)

      columns = ['boxstyle_b1', 'boxstyle_b2']
      for i in columns:
        one_hot_vectors = self.encoder3.transform(df_1[i])
        feature_names = self.encoder3.get_feature_names_out()
        temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
        df_1 = pd.concat([df_1,temp], axis=1)
        df_1.drop(columns=i, inplace=True)

      return df_1


class Data_clusterer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        '''
        Adhiere cluster creado por modelo seleccionado a dataframe base.
        :model: Modelo cluster.
        '''
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        :X: dataframe base.
        :return: df base con columna cluster adherida.
        '''
        clusters = self.model.predict(X)
        clusters = pd.Series(clusters, index=X.index, name='cluster')
        temp = X.merge(clusters, how='outer', left_index=True, right_index=True)
        return temp


class Model_applied(BaseEstimator, TransformerMixin):
  def __init__(self, model):
    '''
    Adhiere predicción de modelo seleccionado a df y probabilidades.
    :model: Modelo clasificación escogido.
    '''
    self.model = model

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    '''
    :X: dataframe base.
    :return: df base con columna predicción y columnas probabilidades.
    '''
    df_pred = pd.DataFrame()
    pred_new = self.model.predict(X)
    probabilities = self.model.predict_proba(X)
    df_pred['boxer1_pred'] = pred_new
    df_pred['prob_win'] = probabilities[:,1]
    df_pred['prob_loss'] = probabilities[:,0]
    return df_pred


def check_fails_and_probas(df_cluster, y_true, y_pred, prob_loss, prob_win, figsize=(5,3)):
  '''
  Función para revisar errores, aciertos y probabilidades de modelo, según sus respectivos cluster.
  :df_cluster: dataframe con columna de clusters.
  :y_true: y verdadero.
  :y_pred: y predicción.
  :prob_loss: probabilidad boxer 1 pierde.
  :prob_win: probabilidad boxer 1 gana.
  :figsize=(5,3): figsize de las gráficas.
  :return: df cluster con columnas true res, pred res, goodpred, prob loss, prob win.
  :plot: conteo de falsos por verdaderos según clusters, histograma de verdaderos y falsos\
         según la probabilidad generada.
  :print: porcentaje de falsos por verdaderos según cada cluster.
  '''
  dfx = df_cluster.copy()
  dfx['true_res'] = y_true.values
  dfx['pred_res'] = y_pred.values
  dfx['goodpred'] = (dfx.true_res == dfx.pred_res).values
  dfx['prob_loss'] = prob_loss.values
  dfx['prob_win'] = prob_win.values

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

  fig,ax=plt.subplots(figsize=figsize)
  sns.histplot(dfx[dfx.goodpred==False].prob_win)
  ax.set_title('Cantidad de False según win pred value')
  plt.show()

  return dfx
