import requests
from lxml import html
from bs4 import BeautifulSoup
import wikipediaapi
import pandas as pd
import numpy as np

def new_box_info(boxer1, boxer2):
  '''
  Make a row with the info needed to get a prediction.
  Need to install wiki api: !pip install wikipedia-api (colab)
  :boxer1: name of the boxer 1.
  :boxer 2: name of the boxer 2.
  '''
  boxer1 = boxer1.str.title()
  boxer2 = boxer2.str.title()
  wiki = wikipediaapi.Wikipedia('english')  # 'en' specifies the language (English in this case)
  # boxer 1
  page = wiki.page(boxer1)
  text_boxer1 = page.text
  summary_boxer1 = page.summary
  len_text_boxer1 = len(text_boxer1)
  len_summary_boxer1 = len(summary_boxer1)
  # boxer 2
  page = wiki.page(boxer2)
  text_boxer2 = page.text
  summary_boxer2 = page.summary
  len_text_boxer2 = len(text_boxer1)
  len_summary_boxer2 = len(summary_boxer1)
