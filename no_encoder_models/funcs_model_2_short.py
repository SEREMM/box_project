from gral_funcs_box_data.py import Data_clusterer, Model_applied
import numpy as np


def feat_eng_2(df):
  '''
  Feature engineering values fro the 2nd (short) df.
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
  df['b1_more_fame'] = np.where(df.len_text_boxer1 > df.len_text_boxer2, 1, -1)

  return df



def model_trainer(X, y, feat_eng_func, cluster, scaler, model):
  '''
  Train cluster, scaler and model for new preds.
  Receives:
    :X: Train data x.
    :y: Train data y.
    :feat_eng_func: Function to feature engineering data.
    :cluster: Cluster to fit.
    :scaler: Scaler to fit.
    :model: Model to fit.
  Returns:
    cluster_trained.
    scaler_trained.
    model_trained.
    x_cluster: Data received with column cluster.
    modelo_objeto: Model_applied object wich can predict and transform data, using transform
      to retreives a df with some columns about prediction.
  '''
  x_0 = feat_eng_func(X)
  cluster.fit(x_0)
  cluster_trained = Data_clusterer(cluster)
  x_cluster = cluster_trained.transform(x_0)
  scaler_trained = scaler.fit(x_cluster)
  x_scaled = scaler_trained.transform(x_cluster)
  model_trained = model.fit(x_scaled, y)
  modelo_objeto = Model_applied(model_trained)
  return cluster_trained, scaler_trained, model_trained, x_cluster, modelo_objeto 


def fiabilidad(col_prob, col_clus):
  fiable = []
  datos = []
  for p, c in zip(col_prob, col_clus):
    if c == 0:
      datos.append('mucha data (1239 data clus, 1672 data total, 74 %)')
      if (p > .75) | (p < .25):
        fiable.append('muy fiable acierta en + de 80 %')
      elif (p > .46) & (p < .55):
        fiable.append('no fiable, acierta en un 50 %')
      else:
        fiable.append('fiable, acierta en + de 70 %')
    
    if c == 1:
      datos.append('poca data (207 data clus, 1672 data total, 12 %)')
      if (p < .25):
        fiable.append('muy fiable acierta en + de 80 %')
      elif (p > .70):
        fiable.append('no medido')
      elif (p > .46) & (p < .55):
        fiable.append('no fiable, acierta en un - 40 %')
      else:
        fiable.append('fiable, acierta en + de 70 %')

    if c == 2:
      datos.append('poca data (189 data clus, 1672 data total, 11 %)')
      if (p < .25):
        fiable.append('no medido')
      elif (p > .65):
        fiable.append('muy fiable, acierta + de 80 %')
      elif (p > .25) & (p < .45):
        fiable.append('no fiable, acierta en - 40 % hasta - 80 %, contrario')
      else:
        fiable.append('fiable, acierta en + de 70 %')

    if c == 3:
      datos.append('muy poca data (19 data clus, 1672 data total, 1 %)')
      fiable.append('nunca se equivoco')

    if c == 4:
      datos.append('muy poca data (18 data clus, 1672 data total, 1 %)')
      if (p < .55):
        fiable.append('no medido')
      elif (p > .65):
        fiable.append('muy fiable, acierta + de 80 %')
      else:
        fiable.append('no fiable, acierta en -.50 a -.80, contrario')

  return fiable, datos


def new_pred_2(X, feature_eng_func, fitted_cluster, fitted_scaler, fitted_model):
  '''
  Function that join the steps to bring a prediction for new data since before feature engineering process, without encoder.
  Parameters:
    :X: First data to predict
    :feature_eng_func: Function to feature engineering.
    :fitted_cluster: Object of class Data_clusterer() initialized and fitted.
    :fitted_scaler: Object of class StandardScaler() [or other scaler] initialized and fitted.
    :fitted_model: Object of class Model_applied() initialized and fitted.
  Returns:
    DataFrame with columns ['boxer1_pred', 'prob_win', 'prob_loss', 'cluster', 'initial_index'].
  '''
  x0 = feature_eng_func(X)
  x_clustered = fitted_cluster.transform(x0)
  x_scaled = fitted_scaler.transform(x_clustered)
  y_pred = fitted_model.transform(x_scaled)
  y_pred['initial_index'] = x_clustered.index
  y_pred = y_pred.merge(x_clustered.cluster, how='left', left_on='initial_index', right_index=True)
  y_pred['reliability'], y_pred['data_amount'] = fiabilidad(y_pred.prob_win, y_pred.cluster)
  y_pred = y_pred[['boxer1_pred','prob_win','cluster','reliability','data_amount','prob_loss','initial_index']]

  return y_pred