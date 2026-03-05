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

YEARS = np.arange(2014, 2027, 1)
HEADS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}

with open("Documents/bracket-bot/data/schools.json", "r") as file:
    SCHOOLS = json.load(file)

SCHOOLS = list(SCHOOLS['schools'])
school_years = list(product(YEARS, SCHOOLS))

def url_to_table(url, year, school):
    if url.endswith('advanced.html'):
        suffix = 'advanced'
    else:
        suffix = 'basic'

    file_name = f'Documents/bracket-bot/data/games/{school}_{year}_{suffix}.csv'
    if Path("my_file.csv").exists():
        print(f"File exists, using cached table for {school} {year}")
        return
    
    response = requests.get(url, headers=HEADS)
    if response.status_code == 429:
        raise RuntimeError("Rate limited (429). Stopping.")
    if response.status_code == 404:
        print(f"404: {url}")
        return
    
    try:
        tables = pd.read_html(response.text)
        df = tables[0]
    except ValueError:
        # No tables
        print(f'No tables for {year}, {school}')
        return

    
    df.to_csv(file_name,index=False)
    return


for i in school_years:
    year, school = str(i[0]), i[1]
    url = f"https://www.sports-reference.com/cbb/schools/{school}/men/{year}-gamelogs.html"
    url_to_table(url, year, school)
    time.sleep(random.uniform(3,3.5))

    url = f"https://www.sports-reference.com/cbb/schools/{school}/men/{year}-gamelogs-advanced.html"
    url_to_table(url, year, school)
    time.sleep(random.uniform(3,3.5))

    print(f'{school} {year} complete.')

