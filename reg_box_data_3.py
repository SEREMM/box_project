import wikipediaapi
import pandas as pd
import numpy as np

def box_data_reg_from_list(boxer1_list, b1_odds_list, draw_odds_list, boxer2_list, b2_odds_list):
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
    if func == 'american':
      lista = odds_american_to_decimal(odds)
    elif func == 'fractional':
      lista = odds_fractional_to_decimal(odds)

    lista_dec_perc = []
    lista_dec_perc.append(round(lista[0] / sum(lista), 2))
    lista_dec_perc.append(round(lista[1] / sum(lista), 2))
    lista_dec_perc.append(round(lista[2] / sum(lista), 2))

    return lista_dec_perc[0], lista_dec_perc[2]

  def convertor_odds(b1_odds, draw_odds, b2_odds):
    b1 = b1_odds
    b2 = b2_odds
    draw = draw_odds
    tipo = input('Odds in american (a), in fractional (f): ')
    lista = [b1,draw,b2]
    if tipo == 'a':
        tipo = 'american'
    elif tipo == 'f':
        tipo = 'fractional'
    b1_bet, b2_bet = function_odds(lista, tipo)
    print(f'\nb1 odds: {b1_bet}, b2 odds: {b2_bet}\n')
    return b1_bet, b2_bet

  def reg_1(texto):
    registro = input(texto)
    return registro

  def reg_2(boxer_name):
    boxer = boxer_name.str.title().str.strip()[0]
    wiki = wikipediaapi.Wikipedia('english')  # 'en' specifies the language (English in this case)
    page = wiki.page(boxer)
    text_boxer = page.text
    summary_boxer = page.summary
    len_text_boxer = len(text_boxer)
    len_summary_boxer = len(summary_boxer)
    print(f'info wiki: {page.title}\n')
    return text_boxer, summary_boxer, len_text_boxer, len_summary_boxer

  df = pd.DataFrame()
  for boxer1, b1_odd, d_odd, boxer2, b2_odd in zip(boxer1_list, b1_odds_list, draw_odds_list, boxer2_list, b2_odds_list):
    continuar = input('Continuar agregando registros (y/n): ')
    if continuar == 'y':
      textos = {'boxer1':'','b1_w':'Number of win boxer1: ',
      'b1_wk':'Number of ko wins of boxer1: ','b1_d':'Number of draws of boxer1: ',
      'b1_l':'Number of loss of boxer1: ','b1_lk':'Number of ko loss of boxer1: ',
      'wiki_boxer1':'',
      'boxer2':'','b2_w':'Number of win boxer2: ',
      'b2_wk':'Number of ko wins of boxer2: ','b2_d':'Number of draws of boxer2: ',
      'b2_l':'Number of loss of boxer2: ','b2_lk':'Number of ko loss of boxer2: ',
      'wiki_boxer2':'', 'odds':''}
  
      data = pd.DataFrame()
      current_key_value = 0
      key_values = list(textos.items())
  
      while current_key_value < len(key_values):
        if (key_values[current_key_value][0] == 'wiki_boxer1'):
          a,b,c,d = reg_2(data['boxer1'])
          data['text_boxer1'], data['summary_boxer1'], data['len_text_boxer1'], data['len_summary_boxer1'] = a,b,c,d
          current_key_value += 1
        elif (key_values[current_key_value][0] == 'wiki_boxer2'):
          a,b,c,d = reg_2(data['boxer2'])
          data['text_boxer2'], data['summary_boxer2'], data['len_text_boxer2'], data['len_summary_boxer2'] = a,b,c,d
          current_key_value += 1
        elif (key_values[current_key_value][0] == 'odds'):
          data['b1_bet'], data['b2_bet'] = convertor_odds(b1_odd, d_odd, b2_odd)
          data['result'] = ''
          data['endtype'] = ''
          data['endround'] = ''
          df = pd.concat([df, data])
          current_key_value += 1
          print('\n' + '=' * 10 + '\n')
        elif (key_values[current_key_value][0] == 'boxer1'):
          b1 = boxer1.lower()
          data['boxer1'] = [b1]
          print(f'boxer 1: {data.boxer1[0]}\n')
          current_key_value += 1
        elif (key_values[current_key_value][0] == 'boxer2'):
          b2 = boxer2.lower()
          data['boxer2'] = [b2]
          print(f'boxer 2: {data.boxer2[0]}\n')
          current_key_value += 1
        else:
          resp = reg_1(key_values[current_key_value][1])
          if resp == 'd':
            current_key_value -= 1
            continue
          elif resp == 'c':
            break
          else:
            data[key_values[current_key_value][0]] = [resp]
            current_key_value += 1
    elif continuar == 'n':
      break
      
  return df[['result', 'endtype', 'endround',
'boxer1','b1_w','b1_wk', 'b1_d', 'b1_l', 'b1_lk', 'b1_bet',
'text_boxer1', 'summary_boxer1','len_text_boxer1', 'len_summary_boxer1',
'boxer2','b2_w', 'b2_wk', 'b2_d', 'b2_l','b2_lk', 'b2_bet',
'text_boxer2', 'summary_boxer2', 'len_text_boxer2','len_summary_boxer2']]
