import pandas as pd
import numpy as np

def function_odds(odds, func='american'):
  '''
  Convert the odds type (american or fractional) into decimal.
  :odds(list): the odds to convert, example: [-500, 2000, 350].
  :func(string, default:american): american, fractional.
  '''
  if func == 'american':
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
        print(f'posicion {count}: valor original: {i}, valor decimal apuesta: {res}\n')
        count += 1
      return lista_dec
    lista = odds_american_to_decimal(odds)
    
  elif func == 'fractional':
    def odds_fractional_to_decimal(odds):
      lista_dec = []
      count = 0
      for i in odds:
        res = (i) + 1
        lista_dec.append(res)
        print(f'posicion {count}: valor original: {i}, valor decimal apuesta: {res}\n')
        count += 1
      return lista_dec
    lista = odds_fractional_to_decimal(odds)

  lista_dec_perc = []
  lista_dec_perc.append(round(lista[0] / sum(lista), 2))
  lista_dec_perc.append(round(lista[1] / sum(lista), 2))
  lista_dec_perc.append(round(lista[2] / sum(lista), 2))

  titulo = 'Valores en proporción para modelo:'
  print(f'\n{titulo}')
  print('=' * len(titulo))
  print(f'boxer1: {lista_dec_perc[0]}, draw: {lista_dec_perc[1]}, boxer2: {lista_dec_perc[2]}')
