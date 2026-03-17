import pandas as pd
import numpy as np
import json

with open("Documents/bracket-bot/data/schools.json", "r") as file:
    SCHOOLS = json.load(file)

SCHOOLS = SCHOOLS['schools']
# Home Away Dedupe

YEARS = np.arange(2021, 2025, 1)

LABEL_COL = 'score_Rslt'
df = pd.read_csv('Documents/bracket-bot/data/years/all_games_2025_with_sos_features.csv')
XCOLS = list(df.columns[157:]) + ['Type']
XCOLS = [x for x in XCOLS if x not in ['team', 'opp_slug']]
df['train_test'] = 'test'

for year in YEARS:
    cur = pd.read_csv(f'Documents/bracket-bot/data/years/all_games_{year}_with_sos_features.csv')
    cur['train_test'] = 'train'
    df = pd.concat([df, cur], axis=0, ignore_index=True)

df = df[df['Gtm'] > 6]

print(df[~df['team'].isin(SCHOOLS)]['team'].unique())
print(sorted(df[~df['opp_slug'].isin(SCHOOLS)]['opp_slug'].unique()))
pd.Series(df[~df['opp_slug'].isin(SCHOOLS)]['opp_slug'].unique()).to_csv('missing_schools.csv', index=False)
pd.Series(SCHOOLS).to_csv('schools.csv',index=False)

#print(df.shape)
before_games = set(list(zip(df['team'], df['opp_slug'], df['Date'])))
df = pd.merge(df, df, how='inner', left_on=['opp_slug', 'Date'], right_on = ['team', 'Date'], suffixes=('_team_a', '_team_b'))
#print(df.columns)
after_games = set(list(zip(df['team_team_a'], df['opp_slug_team_a'], df['Date'])))
dropped_games = pd.DataFrame({'tuple': pd.Series(list(before_games.difference(after_games)))})
dropped_games[['team', 'opp_slug', 'Date']] = pd.DataFrame(dropped_games['tuple'].tolist(), index=dropped_games.index)
dropped_games['team_in_schools'] = dropped_games['team'].isin(SCHOOLS)
dropped_games['opp_in_schools'] = dropped_games['opp_slug'].isin(SCHOOLS)
dropped_games.to_csv('dropped_games.csv',index=False)
#print(dropped_games['team'].value_counts())
#print(dropped_games['opp_slug'].value_counts())
#print(pd.to_datetime(dropped_games['Date']).value_counts(bins=10))

#print(df.shape)










#df_train = df[df['train_test']=='train']
#df_test = df[df['train_test']=='test']
#X_train = df_train[XCOLS]
#X_test= df_test[XCOLS]
#y_train = df_train[LABEL_COL]
#y_test = df_test[LABEL_COL]

#df.to_csv('Documents/bracket-bot/data/train_test_files/unified_train_test.csv', index=False)
#X_train.to_csv('Documents/bracket-bot/data/train_test_files/X_train.csv', index=False)
#X_test.to_csv('Documents/bracket-bot/data/train_test_files/X_test.csv', index=False)
#y_train.to_csv('Documents/bracket-bot/data/train_test_files/y_train.csv', index=False)
#y_test.to_csv('Documents/bracket-bot/data/train_test_files/y_test.csv', index=False)
