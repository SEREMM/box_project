import requests
# from lxml import html
# from bs4 import BeautifulSoup
import wikipediaapi
import pandas as pd
import numpy as np
import csv

file_name = 'manual_box_reg_short.csv'

def odds_american_to_decimal(odds):
  lista_dec = []
  count = 0
  for i in odds:
    if i < 0:
      decimal_odds = 100 / abs(i) + 1
      res = round(decimal_odds, 2)
      lista_dec.append(res)
    elif i > 0:
      decimal_odds = (i / 100) + 1
      res = round(decimal_odds, 2)
      lista_dec.append(res)
    else:
      res = i
      lista_dec.append(res)
    if count == 0:
      print(f'\nValor decimal apuesta a boxer 1: {res}\n')
    elif count == 2:
      print(f'\nValor decimal apuesta a boxer 2: {res}\n')
    count += 1
  return lista_dec

def odds_fractional_to_decimal(odds):
  lista_dec = []
  count = 0
  for i in odds:
    res = (i) + 1
    lista_dec.append(res)
    if count == 0:
      print(f'\nValor decimal apuesta a boxer 1: {res}\n')
    elif count == 2:
      print(f'\nValor decimal apuesta a boxer 2: {res}\n')
    count += 1
  return lista_dec

def function_odds(odds, func='american'):
  '''
  Convert the odds type (american or fractional) into decimal.
  :odds(list): the odds to convert, example: [-500, 2000, 350].
  :func(string, default:american): american, fractional.
  '''
  if func == 'american':
    lista = odds_american_to_decimal(odds)
  elif func == 'fractional':
    lista = odds_fractional_to_decimal(odds)

  lista_dec_perc = []
  lista_dec_perc.append(round(lista[0] / sum(lista), 2))
  lista_dec_perc.append(round(lista[1] / sum(lista), 2))
  lista_dec_perc.append(round(lista[2] / sum(lista), 2))

  return lista_dec_perc[0], lista_dec_perc[2]

def convertor_odds():
  b1 = int(input('Odds boxer1 win: '))
  b2 = int(input('Odds boxer2 win: '))
  draw = int(input('Odds draw: '))
  tipo = input('Odds in american (a), in fractional (f): ')
  lista = [b1,draw,b2]
  if tipo == 'a':
      tipo = 'american'
  elif tipo == 'f':
      tipo = 'fractional'
  b1_bet, b2_bet = function_odds(lista, tipo)
  print(f'\nb1 odds: {b1_bet}, b2 odds: {b2_bet}\n')
  return b1_bet, b2_bet


class Reg_box_data():
  def reg_1(self, texto):
    registro = input(texto)
    return registro

  def reg_2(self, boxer_name, texto):
    boxer = boxer_name.str.title().str.strip()[0]
    wiki = wikipediaapi.Wikipedia('english')  # 'en' specifies the language (English in this case)
    page = wiki.page(boxer)
    text_boxer = page.text
    summary_boxer = page.summary
    len_text_boxer = len(text_boxer)
    len_summary_boxer = len(summary_boxer)
    print(f'info: {page.title}\n')
    return text_boxer, summary_boxer, len_text_boxer, len_summary_boxer

  def registro(self):
    while True:
      textos = {'result':'Boxer 1 result (win, loss): ',
      'endtype':'End by ko or decision or rtd: ','endround':'End round: ',
      'boxer1':'Name of boxer1: ','b1_w':'Number of win boxer1: ',
      'b1_wk':'Number of ko wins of boxer1: ','b1_d':'Number of draws of boxer1: ',
      'b1_l':'Number of loss of boxer1: ','b1_lk':'Number of ko loss of boxer1: ',
      'wiki_boxer1':'Info wiki boxer1',
      'boxer2':'Name of boxer2: ','b2_w':'Number of win boxer2: ',
      'b2_wk':'Number of ko wins of boxer2: ','b2_d':'Number of draws of boxer2: ',
      'b2_l':'Number of loss of boxer2: ','b2_lk':'Number of ko loss of boxer2: ',
      'wiki_boxer2':'Info wiki boxer2', 'odds':'Ingrese las apuestas: '}

      data = pd.DataFrame()
      current_key_value = 0
      key_values = list(textos.items())

      while current_key_value < len(key_values):
        if (key_values[current_key_value][0] == 'wiki_boxer1'):
          a,b,c,d = self.reg_2(data['boxer1'], key_values[current_key_value][1])
          data['text_boxer1'], data['summary_boxer1'], data['len_text_boxer1'], data['len_summary_boxer1'] = a,b,c,d
          current_key_value += 1
        elif (key_values[current_key_value][0] == 'wiki_boxer2'):
          a,b,c,d = self.reg_2(data['boxer2'], key_values[current_key_value][1])
          data['text_boxer2'], data['summary_boxer2'], data['len_text_boxer2'], data['len_summary_boxer2'] = a,b,c,d
          current_key_value += 1
        elif (key_values[current_key_value][0] == 'odds'):
          data['b1_bet'], data['b2_bet'] = convertor_odds()
          current_key_value += 1
        else:
          resp = self.reg_1(key_values[current_key_value][1])
          if resp == 'd':
            current_key_value -= 1
            continue
          elif resp == 'c':
            break
          else:
            data[key_values[current_key_value][0]] = [resp]
            current_key_value += 1

      return data[['result', 'endtype', 'endround',
    'boxer1','b1_w','b1_wk', 'b1_d', 'b1_l', 'b1_lk', 'b1_bet',
    'text_boxer1', 'summary_boxer1','len_text_boxer1', 'len_summary_boxer1',
    'boxer2','b2_w', 'b2_wk', 'b2_d', 'b2_l','b2_lk', 'b2_bet',
    'text_boxer2', 'summary_boxer2', 'len_text_boxer2','len_summary_boxer2']]


def create_csv_file():
    df = pd.DataFrame(columns=['result', 'endtype', 'endround',
    'boxer1','b1_w','b1_wk', 'b1_d', 'b1_l', 'b1_lk', 'b1_bet',
    'text_boxer1', 'summary_boxer1','len_text_boxer1', 'len_summary_boxer1',
    'boxer2','b2_w', 'b2_wk', 'b2_d', 'b2_l','b2_lk', 'b2_bet',
    'text_boxer2', 'summary_boxer2', 'len_text_boxer2','len_summary_boxer2'])
    df.to_csv(f'{file_name}', index=False)

def append_csv_row(row):
    with open(f'{file_name}', mode='a', newline='') as file:
        writer = csv.writer(file)
        csv_row = [s.encode('ascii', 'ignore').decode() if isinstance(s, str) else s for s in row]
        writer.writerow(csv_row)

def delete_and_repeat(titulo):
    try:
      df = pd.read_csv(f'{file_name}')
      last_row = df.iloc[-1:,:][['boxer1','boxer2']]
      df = df.drop(df.index[-1])
      df.to_csv(f'{file_name}', index=False)
      print(f"Deleted row: {last_row}")
    except FileNotFoundError:
        print("--- No previous row to delete ---")
    except UnicodeDecodeError:
        df = pd.read_csv(f'{file_name}', encoding='latin_1')
        last_row = df.iloc[-1:,:][['boxer1','boxer2']]
        df = df.drop(df.index[-1])
        df.to_csv(f'{file_name}', index=False)
        print(f"Deleted row: {last_row}")


def start():
    option = ''
    while option != 'c':
        titulo = '"Enter" para ingresar registro, "c" para cerrar, "del" para eliminar el registro anterior,'
        titulo2 = '\n"d" durante los registros para regresar al anterior: '
        print('=' * len(titulo) + '\n')
        print(titulo + titulo2 + '\n')
        option = input()
        print('=' * len(titulo) + '\n')
        if option == '':
            temp = Reg_box_data().registro()
            val_shape = temp.values.shape[1]
            csv_row = temp.values.reshape(val_shape)
            append_csv_row(csv_row)
        elif option == 'c':
            print("Programa finalizado.")
        elif option == 'del':
            delete_and_repeat('Eliminar fila')
        else:
            print("--- Opción no válida ---")

# main
try:
    df = pd.read_csv(f'{file_name}')
except FileNotFoundError:
    create_csv_file()
except UnicodeDecodeError:
    df = pd.read_csv(f'{file_name}', encoding='latin_1')

start()
