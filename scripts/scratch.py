import pandas as pd 

df = pd.read_csv('Documents/bracket-bot/data/years/games_2025.csv')
lst = df.columns
pd.DataFrame({'cols':lst}).to_csv('full_columns.csv')