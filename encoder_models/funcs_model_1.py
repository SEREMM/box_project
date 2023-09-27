import pandas as pd
import numpy as np

def feat_eng_1(df):
  '''
  Feature engineering values for the first df.
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


class Features_encoder_1(BaseEstimator, TransformerMixin):
    def __init__(self, encoder1=0, encoder2=0, encoder3=0):
      '''
      Vectorizador de string features del 1er training df.
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
    

def new_pred_1(X, feature_eng_func, fitted_encoder, fitted_cluster, fitted_scaler, fitted_model):
  '''
  Function that join the steps to bring a prediction for new data since before feature engineering process, with encoder.
  Parameters:
    :X: First data to predict.
    :feature_eng_func: Function to feature engineering.
    :fitted_encoder: Object of class Features_encoder() initialized and fitted.
    :fitted_cluster: Object of class Data_clusterer() initialized and fitted.
    :fitted_scaler: Object of class StandardScaler() [or other scaler] initialized and fitted.
    :fitted_model: Object of class Model_applied() initialized and fitted.
  Returns:
    DataFrame with columns .
  '''
  x0 = feature_eng_func(X)
  x1 = fitted_encoder.transform(x0)
  x_clustered = fitted_cluster.transform(x1)
  x_scaled = fitted_scaler.transform(x_clustered)
  y_pred = fitted_model.transform(x_scaled)
  y_pred['initial_index'] = x_clustered.index
  y_pred = y_pred.merge(x_clustered.cluster, how='left', left_on='initial_index', right_index=True)  
  y_pred = y_pred[['boxer1_pred','prob_win','cluster','prob_loss','initial_index']]

  return y_pred