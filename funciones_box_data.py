import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


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


def feat_eng(df):
  '''
  Feature engineering values.
  :df: Dataframe sobre el cual agregar las características.
  :return: df con las características.
  '''
  df = df.copy()
  suma_peleas_b1 = df.b1_w + df.b1_d + df.b1_l
  suma_peleas_b2 = df.b2_w + df.b2_d + df.b2_l
  df['b1_momios_menores'] = np.where(df.b1_bet < df.b2_bet, 1, 0)
  df['b1_bet_inversa'] = abs(df.b1_bet - 1)
  df['b2_bet_inversa'] = abs(df.b2_bet - 1)
  df = df.drop(columns=['b1_bet','b2_bet'])
  df['b1_mas_peleas'] = np.where(suma_peleas_b1 > suma_peleas_b2, 1, 0)
  df['b1_menos_peleas'] = np.where(suma_peleas_b1 < suma_peleas_b2, -1, 0)
  df['b1_mas_win_perc'] = np.where((df.b1_w / suma_peleas_b1) > (df.b2_w / suma_peleas_b2), 1, 0)
  df['b1_menos_win_perc'] = np.where((df.b1_w / suma_peleas_b1) < (df.b2_w / suma_peleas_b2), -1, 0)
  df['b1_menos_loss_perc'] = np.where((df.b1_l / suma_peleas_b1) < (df.b2_l / suma_peleas_b2), 1, 0)
  df['b1_mas_loss_perc'] = np.where((df.b1_l / suma_peleas_b1) > (df.b2_l / suma_peleas_b2), -1, 0)
  df['b1_mas_ko_perc'] = np.where((df.b1_wk / df.b1_w) > (df.b2_wk / df.b2_w), 1, 0)
  df['b1_menos_ko_perc'] = np.where((df.b1_wk / df.b1_w) < (df.b2_wk / df.b2_w), -1, 0)
  df['b1_invicto'] = np.where(df.b1_l <= 0, 1, 0)
  df['b2_invicto'] = np.where(df.b2_l <= 0, -1, 0)
  df['b1_local'] = np.where(df.c_f == df.region_b1, 1, 0)
  df['b2_local'] = np.where(df.c_f == df.region_b2, -1, 0)

  return df


from sklearn.base import BaseEstimator, TransformerMixin


class Features_encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder1=0, encoder2=0, encoder3=0):
      '''
      Vectorizador de string features.
        :encoder1(defaul 0): vectorizador regiones [c_f, region_b1, region_b2] (if pass encoder, else omit).
        :encoder2(default 0): vectorizador posturas [stance_b1, stance_b2] (if pass encoder, else omit).
        :encoder3(default 0): vectorizador estilos [boxstyle_b1, boxstyle_b2] (if pass encoder, else omit).
      Returns: Dataframe with vectorized values
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
      if self.encoder1 == 0:
        print('No encoder 1\n')
      else:
        columns = ['c_f', 'region_b1', 'region_b2']
        for i in columns:
          one_hot_vectors = self.encoder1.transform(df_1[i])
          feature_names = self.encoder1.get_feature_names_out()
          temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
          df_1 = pd.concat([df_1,temp], axis=1)
          df_1.drop(columns=i, inplace=True)

      if self.encoder2 == 0:
        print('No encoder 2\n')
      else:
        columns = ['stance_b1', 'stance_b2']
        for i in columns:
          one_hot_vectors = self.encoder2.transform(df_1[i])
          feature_names = self.encoder2.get_feature_names_out()
          temp = pd.DataFrame(one_hot_vectors.toarray(), columns=feature_names, index=df_1.index)
          df_1 = pd.concat([df_1,temp], axis=1)
          df_1.drop(columns=i, inplace=True)

      if self.encoder3 == 0:
        print('No encoder 3\n')
      else:
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
  :plot: conteo de falsos por verdaderos según clusters, Perc false / total by Prob win, Perc. false / total for the clusters.
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

  #fig,ax=plt.subplots(figsize=figsize)
  #sns.histplot(dfx[dfx.goodpred==True].prob_win)
  #sns.histplot(dfx[dfx.goodpred==False].prob_win)
  #ax.set_title('Comparacion win / false pred value')
  #plt.show()

  a = round(dfx.prob_win,1)
  b = dfx[['goodpred','prob_win']]
  b['prob_win'] = a
  b = pd.get_dummies(b, columns=['goodpred'])
  b = b.groupby('prob_win').sum()
  b['false_over_total'] = round(b.goodpred_False / (b.goodpred_False + b.goodpred_True), 2)

  fig, ax = plt.subplots(figsize=figsize)
  sns.barplot(x=b.index, y=b.false_over_total)
  ax.set_ylim(0,1)
  plt.ylabel('Perc. false over total')
  plt.xlabel('Prob. win ex.(0.5 = from 0.46 to 0.55)')
  plt.title('Perc false / total by Prob win')
  plt.show()

  clusters = dfx.cluster.unique()
  for i in clusters:
    temp1 = dfx.copy()
    temp1 = temp1[['goodpred','cluster','prob_win']].query(f'cluster == {i}')
    temp1['prob_win'] = round(temp1.prob_win, 1)
    temp1 = pd.get_dummies(temp1, columns=['goodpred'])
    temp1 = temp1.groupby(['cluster','prob_win']).sum().reset_index()
    temp1['false_over_total'] = temp1.goodpred_False / (temp1.goodpred_True + temp1.goodpred_False)
  
    fig,ax = plt.subplots(figsize=(figsize))
    sns.barplot(data=temp1, x='prob_win', y='false_over_total')
    ax.set_ylim(0,1)
    plt.xlabel('Prob. Win ex.(0.5 = from 0.46 to 0.55)')
    plt.ylabel('Perc. false over total')
    plt.title(f'Perc. false / total. Cluster {i}')
    plt.show()
  
  return dfx


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def plot_clusters(clusters, dim_reduct_values, cmap="Set2", figsize=(5,3)):
    """
    Plots the projection of the features colored by clusters.

    Parameters:
        clusters (numpy array): The clusters of the data.
        dim_reduct_values (numpy array): The dimensionality reducted values of features.
        cmap (str or colormap, optional): The colormap for coloring clusters. Default is "Set2".
        figsize (tuple, optional): The size of the plot. default is (5,3).

    Returns:
        None (displays the plot).
    """
    cmap = plt.get_cmap(cmap)

    n_clusters = np.unique(clusters).shape[0]

    fig = plt.figure(figsize=figsize)
    # Plot the dimensionality-reduced features on a 2D plane
    plt.scatter(dim_reduct_values[:, 0], dim_reduct_values[:, 1],
                c=[cmap(x/n_clusters) for x in clusters], s=40, alpha=.4)
    plt.title('dim reduct projection of values, colored by clusters', fontsize=14)
    plt.show()


def find_optimal_clusters(data, scaler=StandardScaler(), max_clusters=10, clustering_model=KMeans, figsize=(5,3)):
    """
    Function to find the optimal number of clusters using the Elbow Method.

    Parameters:
        data (numpy.ndarray or pandas.DataFrame): The dataset to be analyzed.
        scaler (optional): The scaler for the values. Default is StandardScaler.
        max_clusters (int, optional): The maximum number of clusters to consider.
        clustering_model (optional): The clustering model to use. Default is KMeans.
        figsize (tuple, optional): The size of the plot. Default is (5,3).

    Returns:
        None (plots the Elbow Method graph).
    """
    # Standardize the data to have zero mean and unit variance
    standardized_data = scaler.fit_transform(data)

    # Initialize an empty list to store the within-cluster sum of squares
    wcss = []

    # Calculate WCSS for different number of clusters from 1 to max_clusters
    for num_clusters in range(1, max_clusters + 1):
        model = clustering_model(n_clusters=num_clusters, random_state=42)
        model.fit(standardized_data)
        wcss.append(model.inertia_)  # Sum of squared distances to the closest cluster center

    # Plot the Elbow Method graph
    plt.figure(figsize=figsize)
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method to Find Optimal Number of Clusters')
    plt.xticks(np.arange(1, max_clusters + 1))
    plt.grid()
    plt.show()
