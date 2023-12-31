# -*- coding: utf-8 -*-
"""clases_predictors.ipynb

Automatically generated by Colaboratory.
"""

!pip install dill
!pip install lime

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
import joblib
import dill
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime
from google.colab import userdata
# %run "{userdata.get('gral_funcs_box_data')}"
# %run "{userdata.get('funcs_model_2_short')}"
# %run "{userdata.get('pipeline_cleaner_raw_data')}"
# %xmode #verbose
import warnings
import pdb
    # pdb.set_trace()
warnings.filterwarnings('ignore')

# cluster = joblib.load('/content/cluster_object_kmeans_4_clus.pkl')
# scaler = joblib.load('/content/scaler_standard.pkl')
# model = joblib.load('/content/model_object_svc_.pkl')
# with open('/content/lime_explainer_svc.dill', 'rb') as file:
#   explainer = dill.load(file)

# pd.set_option('display.max_columns', 50)
# # lectura de la data nueva
# bo = pd.read_csv(userdata.get('boxers_info'))
# fi = pd.read_csv(userdata.get('fight_info_real'))
# fr = pd.read_csv(userdata.get('fight_results_real'))
# # merge and split new data
# ndf, new_id_df, new_other_cols = box_data_merger_spliter(bo,fi,fr)

# # separate ndf to get the columns with res nan values
# trial = ndf[ndf.result.isna()]

"""Make predictions"""

class Model_predictions:
  '''
  Class to make predictions to first df received from box_data_merger_spliter() function.
  Warning:
    It should be runned data_column_sep() and new_preds() functions.
  '''
  def __init__(self, cluster_obj, scaler_obj, model_obj, explainer):
    '''
    Constructor method.
    Parameters:
      cluster_obj: Cluster object fitted.
      scaler_obj: Scaler object fitted.
      model_obj: Model object fitted.
      explainer: Lime tabular explainer fitted.
    '''
    self.model = model_obj
    self.cluster = cluster_obj
    self.scaler = scaler_obj
    self.explainer = explainer

  def predict(self, data, id_df):
    res_new, pelea_new, X_new, y_new = data_column_sep(data) # real data
    xn_clus, xn_fe, xn_scal, yn_pred = new_preds(X_new, self.cluster, self.scaler, self.model)
    df1, df2, df3 = check_predictions(res_new, pelea_new, yn_pred, id_df, xn_clus)
    return df3, xn_scal

  def confidence_check(self, data):
    df = data.copy()
    mes1_list = []
    mes2_list = []
    for i, e in zip(df.cluster, df.prob_win):
      if i == 0:
        mes2 = 'fail 19 % of 1150 (val, test)'
        if (e > .35) & (e < .45):
          mes1 = 'warning: .35 to .45, fail 30 %'
        elif (e > .45) & (e < .65):
          mes1 = 'WARNING: .45 to .65, fail 50 %'
        elif (e > .15) & (e < .35):
          mes1 = 'WARNING: .15 to .35, fail 50 %'
        else:
          mes1 = ''
      if i == 1:
        mes2 = 'fail 4 % of 90 (warning: few data)(val, test)'
        if (e > .25) & (e < .35):
          mes1 = 'WARNING: .25 to .35, fail 60 %'
        else:
          mes1 = ''
      if i == 2:
        mes2 = 'not fail, WARNING: only 25 (val, test)'
        mes1 = ''
      if i == 3:
        mes2 = 'fail 10 % of 285 (warning: few data)(val, test)'
        if (e > .45) & (e < .56):
          mes1 = 'warning: .45 to .56, fail 38 %'
        elif (e > .25) & (e < .45):
          mes1 = 'WARNING: .25 to .45, fail 50 %'
        else:
          mes1 = ''
      mes2_list.append(mes2)
      mes1_list.append(mes1)
    df['data_cuant'] = mes2_list
    df['confidence'] = mes1_list
    return df

  def select_columns(self, data):
    df = data.copy()
    df = df[['initial_index', 'fight_id', 'result', 'endround', 'endtype',
            'cluster', 'boxer1_pred', 'goodpred', 'prob_win', 'data_cuant', 'confidence',
            'boxer1', 'boxer2', 'dif_bet', 'dif_len_text',
            'dif_edad', 'dif_height', 'dif_w', 'dif_wk_perc',
            'dif_l', 'dif_lk_perc', 'modelo', 'b1_id', 'b2_id']]
    return df

  def decision_table(self, data):
    df = data.copy()
    df['dif_edad'] = [f'{i} - {e} = {i - e}' for i,e in zip(df.birth_b1, df.birth_b2)]
    df['dif_height'] = [f'{i} - {e} = {i - e}' for i,e in zip(df.height_b1, df.height_b2)]
    df['dif_w'] = [f'{i} - {e} = {i - e}' for i,e in zip(df.b1_w, df.b2_w)]
    df['dif_wk_perc'] = [f'{round(i/ie, 2)} - {round(e/ei,2)} = {round((i/ie) - (e/ei), 2)}'\
                              if (ie > 0) & (ei > 0) else f'{i} - {e} = {i - e}'\
                              for i,e,ie,ei in zip(df.b1_wk, df.b2_wk, df.b1_w, df.b2_w)]
    df['dif_l'] = [f'{i} - {e} = {i - e}' for i,e in zip(df.b1_l, df.b2_l)]
    df['dif_lk_perc'] = [f'{round(i/ie, 2)} - {round(e/ei,2)} = {round((i/ie) - (e/ei), 2)}'\
                              if (ie > 0) & (ei > 0) else f'{i} - {e} = {i - e}'\
                              for i,e,ie,ei in zip(df.b1_lk, df.b2_lk, df.b1_l, df.b2_l)]
    df['dif_bet'] = [f'{round(i, 2)} - {round(e, 2)} = {round(i - e, 2)}' for i,e in zip(df.b1_bet, df.b2_bet)]
    df['dif_len_text'] = [f'{i} - {e} = {round(i - e, 2)}' for i,e in zip(df.len_text_boxer1, df.len_text_boxer2)]
    df['modelo'] = str(self.model.model)
    temp = self.confidence_check(df)
    temp = self.select_columns(temp)
    return temp

  def lime_explanation(self, instance):
    explanation = self.explainer.explain_instance(instance, self.model.model.predict_proba, num_features=10)
    dictio = {}
    for feature, weight in explanation.as_list():
      dictio[f'{feature}'] = round(weight, 4)
    return dictio

  def loop_add_explanation(self, x_values):
    lista = []
    for row in x_values:
      res = self.lime_explanation(row)
      lista.append(res)
    return lista

  def make_prediction(self, data_to_predict, id_df):
    '''
    Method to make predictions for the given data.
    Parameters:
      data_to_predict: First dataframe returned from box_data_merger_splitter() function.
    Return:
      temp: Dataframe with predictions from the model, analysis, and explanation of the prediction.
    '''
    df = data_to_predict.copy()
    temp, x_scal= self.predict(df, id_df)
    temp = self.decision_table(temp)
    temp['explanation'] = self.loop_add_explanation(x_scal)
    temp['date_model_pred'] = datetime.now()
    return temp

# predictor = Model_predictions(cluster, scaler, model, explainer)
# trial2 = predictor.make_prediction(trial, new_id_df)

# trial2.head(2)

"""Predictions logger"""

class CloseException(Exception):
  pass

class Predictions_logger:
  '''
  Class to add model and self predictions to a csv file, in case the file
  doesn't exist it will create it.
  '''
  def __init__(self, file_):
    '''
    Contructor method.
    Parameters:
      file_: filename to record the data, ext: .csv.
    '''
    self.file_ = file_

  def closer(self, message):
      res = input(f'{message}')
      if res == 'c':
        raise CloseException()
      else:
        return res

  def read_file(self):
    '''
    Method for read the file where the model and self predictions are logged.
    Returns:
      df: Dataframe with the mentioned information.
    '''
    try:
      df = pd.read_csv(self.file_).sort_values('fight_id')
      return df
    except KeyError:
      df = pd.read_csv(self.file_)
      return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
      df = pd.DataFrame()
      df.to_csv(self.file_, index=False)
      return df

  def authorization(self, message):
    autho = input(f'{message} (y/n): ')
    if autho == 'y':
      return 1
    else:
      return 0

  def save_file(self, data):
    data.to_csv(self.file_, index=False)
    print('Saved data')

  def display_as_dict(self, data):
    temp = data.to_dict(orient='records')[0]
    keys_to_include = ['cluster', 'boxer1_pred', 'prob_win', 'data_cuant', 'confidence']
    sliced_dict = {key: temp[key] for key in keys_to_include}
    print(sliced_dict)
    keys_to_include = ['boxer1', 'boxer2', 'dif_bet']
    sliced_dict = {key: temp[key] for key in keys_to_include}
    print(sliced_dict)
    keys_to_include = ['dif_len_text', 'dif_edad', 'dif_height']
    sliced_dict = {key: temp[key] for key in keys_to_include}
    print(sliced_dict)
    keys_to_include = ['dif_w', 'dif_wk_perc', 'dif_l', 'dif_lk_perc']
    sliced_dict = {key: temp[key] for key in keys_to_include}
    print(sliced_dict)
    keys_to_include = ['modelo', 'explanation']
    sliced_dict = {key: temp[key] for key in keys_to_include}
    print(sliced_dict)
    print()

  def log_reg(self, data):
    self.display_as_dict(data)
    data['self_pred'] = self.closer('Opinion personal (1 - boxer1 win, 0 - boxer1 loss, c - close): ')
    data['date_self_pred'] = datetime.now()
    df = self.read_file()
    df = pd.concat([df, pd.DataFrame(data)])
    df = df[['fight_id', 'self_pred', 'cluster', 'boxer1_pred', 'prob_win', 'data_cuant', 'confidence',
    'boxer1', 'boxer2', 'dif_bet', 'dif_len_text', 'dif_edad', 'dif_height', 'dif_w', 'dif_wk_perc',
    'dif_l', 'dif_lk_perc', 'modelo', 'b1_id', 'b2_id', 'explanation','date_model_pred','date_self_pred']]
    self.save_file(df)
    print()

  def check_existent_values(self, row, data):
    try:
      id = row.fight_id.values[0]  # Extract the fight_id value from the row
      matching_rows = data[data['fight_id'] == id]  # Find matching rows
      if not matching_rows.empty:  # Check if matching rows exist
        return matching_rows  # Return matching rows
      else:
        return None  # No matching rows found, return 0 or an appropriate value
    except AttributeError:
      return None  # Handle the case where 'row' or 'data' is not a DataFrame
    except KeyError:
      return None

  def record_data_selector(self, data, id='new'):
    if id == 'new':
      try:
        check = self.read_file()
        data = data.drop(data[data.fight_id.isin(check.fight_id.values)].index)
      except AttributeError:
        data = data
    elif id == 'all':
      data = data
    else:
      data = data[data.fight_id == id]
    return data

  def record_data(self, data):
    '''
    Method for record data received from Model_predictions().make_prediction(),
    adding a self prediction.
    Parameters:
      data: Dataframe from the mentioned object in the description.
    '''
    df = data.copy()
    df = self.record_data_selector(df)
    df['prob_win'] = round(df.prob_win, 3)
    df = df[['fight_id', 'cluster', 'boxer1_pred', 'prob_win', 'data_cuant', 'confidence',
        'boxer1', 'boxer2', 'dif_bet', 'dif_len_text', 'dif_edad', 'dif_height',
        'dif_w', 'dif_wk_perc', 'dif_l', 'dif_lk_perc', 'modelo', 'b1_id', 'b2_id',
        'explanation','date_model_pred']]
    size_rows_df = df.shape[0]
    archivo = self.read_file()
    try:
      for i in range(0, size_rows_df):
        temp = df.iloc[i:i+1, :]
        res = self.check_existent_values(temp, archivo)
        if res is not None:
          if res.empty:
            self.log_reg(temp)
          else:
            print('Already exist a record:\n')
            self.display_as_dict(res)
            autho = input("Want to log another one (y/n): ")
            print()
            if autho == 'y':
              self.log_reg(temp)
            else:
              print('siguiente\n')
              continue
        else:
          self.log_reg(temp)
    except CloseException:
        print('Finalizado')

  def delete_row(self, id):
    '''
    Method for delete row.
    Parameters:
      id: Fight id row to delete.
    '''
    try:
      data = pd.read_csv(self.file_)# checar no de problemas el archivo del cual borrar
      filtered_df = data[data['fight_id'] != id]
      delete_row = data[data['fight_id'] == id]
      aut = self.authorization(f'Proced to delete record\n{delete_row.to_dict(orient="records")[0]}\n')
      if aut == 1:
        self.save_file(filtered_df)
        print(f'Row deleted successfully\n')
      else:
        print('Authorization to delete denied\n')
    except Exception as e:
      print(f"Error deleting row: {str(e)} - maybe id mismatch\n")

# logger = Predictions_logger('prueba.csv')

# logger.record_data(trial2)

# logger.read_file()

# logger.delete_row(64)
