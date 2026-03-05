import requests
import pandas as pd
import numpy as np
import time
import random
import json
from itertools import product
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

url = "https://www.sports-reference.com/stathead/basketball/cbb/team-game-finder.cgi?request=1&timeframe=seasons&year_min=2015&comp_type=reg&comp_id=NCAAM"
#https://www.sports-reference.com/stathead/basketball/cbb/team-game-finder.cgi?request=1&comp_type=reg&comp_id=NCAAM&order_by=pts&game_status=1&timeframe=seasons&match=team_game&year_min=2015&offset=200

HEADS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}

response = requests.get(url, headers=HEADS)
print(response.text)
tables = pd.read_html(response.text)
df = tables[0]
print(df)