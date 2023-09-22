import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


from sklearn.model_selection import cross_validate, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def cross_val(model, X, y, cv=5, figsize=(5,3)):
  '''
  Function to cross val ml models and check results.
  Parameters:
    :model: ml model.
    :X: predictor data.
    :y: data to predict.
    :cv: cross val folders, default=5.
    :figsize: Size of the plots, default=(5,3).
  Returns:
    :print: métricas de resultados.
    :plot: Confusion matrix, roc curve, precision recall curve, learning curve.
  '''
  cv_results = cross_validate(model, X, y, cv=cv)

  print(f'Fit time mean: {cv_results["fit_time"].mean()}')
  print(f'Score time mean: {cv_results["score_time"].mean()}')
  print(f'Test score: {cv_results["test_score"]}')
  print(f'Test mean score: {cv_results["test_score"].mean()}')

  y_pred = cross_val_predict(model, X, y, cv=cv)
  conf_matrix = confusion_matrix(y, y_pred)
  plt.figure(figsize=figsize)
  sns.heatmap(conf_matrix, annot=True, fmt='d')
  plt.title('Confusion Matrix')
  plt.ylabel('True')
  plt.xlabel('Predicted')
  plt.show()

  fpr, tpr, _ = roc_curve(y, y_pred)
  roc_auc = auc(fpr, tpr)
  plt.figure(figsize=figsize)
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc='lower right')
  plt.show()

  precision, recall, _ = precision_recall_curve(y, y_pred)
  plt.figure(figsize=figsize)
  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')
  plt.show()

  train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  plt.figure(figsize=figsize)
  plt.title("Learning Curve")
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  plt.grid()
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.1, color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
           label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
           label="Cross-validation score")
  plt.legend(loc="best")
  plt.show()


from sklearn.base import BaseEstimator, TransformerMixin


class Data_clusterer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        '''
        Adhiere cluster creado por modelo seleccionado a dataframe base.
        Parameters.
          :model: Modelo cluster.
        Returns:
          None
        '''
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        Use the cluster passed to predict.
        Parameters:
          :X: dataframe base.
        Returns:
          Df base con columna cluster adherida.
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


def new_pred_1(X, fitted_encoder, fitted_cluster, fitted_scaler, fitted_model):
  '''
  Function that join the steps to bring a prediction for new data since before feature engineering process, with encoder.
  Parameters:
    :X: First data to predict
    :fitted_encoder: Object of class Features_encoder() initialized and fitted.
    :fitted_cluster: Object of class Data_clusterer() initialized and fitted.
    :fitted_scaler: Object of class StandardScaler() [or other scaler] initialized and fitted.
    :fitted_model: Object of class Model_applied() initialized and fitted.
  Returns:
    DataFrame with columns ['boxer1_pred', 'prob_win', 'prob_loss', 'cluster', 'initial_index'].
  '''
  x0 = feat_eng(X)
  x1 = fitted_encoder.transform(x0)
  x_clustered = fitted_cluster.transform(x1)
  x_scaled = fitted_scaler.transform(x_clustered)
  y_pred = fitted_model.transform(x_scaled)
  y_pred['initial_index'] = x_clustered.index
  y_pred = y_pred.merge(x_clustered.cluster, how='left', left_on='initial_index', right_index=True)
  y_pred = y_pred[['boxer1_pred','prob_win','cluster','prob_loss','initial_index']]

  return y_pred


def new_pred_2(X, fitted_cluster, fitted_scaler, fitted_model):
  '''
  Function that join the steps to bring a prediction for new data since before feature engineering process, without encoder.
  Parameters:
    :X: First data to predict
    :fitted_encoder: Object of class Features_encoder() initialized and fitted.
    :fitted_cluster: Object of class Data_clusterer() initialized and fitted.
    :fitted_scaler: Object of class StandardScaler() [or other scaler] initialized and fitted.
    :fitted_model: Object of class Model_applied() initialized and fitted.
  Returns:
    DataFrame with columns ['boxer1_pred', 'prob_win', 'prob_loss', 'cluster', 'initial_index'].
  '''
  x0 = feat_eng_2(X)
  x1 = x0 # fitted_encoder.transform(x0)
  x_clustered = fitted_cluster.transform(x1)
  x_scaled = fitted_scaler.transform(x_clustered)
  y_pred = fitted_model.transform(x_scaled)
  y_pred['initial_index'] = x_clustered.index
  y_pred = y_pred.merge(x_clustered.cluster, how='left', left_on='initial_index', right_index=True)
  y_pred = y_pred[['boxer1_pred','prob_win','cluster','prob_loss','initial_index']]

  return y_pred


def check_fails_and_probas(df_cluster, y_true, y_pred, prob_loss, prob_win, figsize=(5,3)):
  '''
  Función para revisar errores, aciertos y probabilidades de modelo, según sus respectivos cluster.
  Parameters:
    :df_cluster: dataframe con columna de clusters.
    :y_true: y verdadero.
    :y_pred: y predicción.
    :prob_loss: probabilidad boxer 1 pierde.
    :prob_win: probabilidad boxer 1 gana.
    :figsize=(5,3): figsize de las gráficas.
  Returns:
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
    try:
        temp1['false_over_total'] = temp1.goodpred_False / (temp1.goodpred_True + temp1.goodpred_False)

        fig,ax = plt.subplots(figsize=(figsize))
        sns.barplot(data=temp1, x='prob_win', y='false_over_total')
        ax.set_ylim(0,1)
        plt.xlabel('Prob. Win ex.(0.5 = from 0.46 to 0.55)')
        plt.ylabel('Perc. false over total')
        plt.title(f'Perc. false / total. Cluster {i}')
        plt.show()
    except AttributeError:
        print(f'\n::: Attribute error in cluster: {i} :::\n')

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
