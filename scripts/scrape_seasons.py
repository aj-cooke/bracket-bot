import requests
import pandas as pd
import numpy as np
import time
import random

YEARS = np.arange(2014, 2027, 1)
HEADS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}


# Every team for a season
SEASON_BASE = "https://www.sports-reference.com/cbb/seasons/men/"
SEASON_STUBS = {'basic': "-school-stats.html",
                'opp': "-opponent-stats.html",
                'adv': "-advanced-school-stats.html",
                'opp_adv': "-advanced-opponent-stats.html"}

def url_to_table(url):
    response = requests.get(url, headers=HEADS)
    return pd.read_html(response.text)[0]

for year in YEARS:
    for k, v in SEASON_STUBS.items():
        url = f'{SEASON_BASE}{str(year)}{v}'
        df = url_to_table(url)
        print(df)
        df.to_csv(f'Documents/bracket-bot/data/{k}_{str(year)}.csv', index=False)
        time.sleep(random.uniform(0.5,2.0))
        print(f'{str(year)} {k} completed')

    




