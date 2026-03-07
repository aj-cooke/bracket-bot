import pandas as pd
import numpy as np

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
# df = df[df['site'].fillna('H') != '@']
#df['t1'] = np.where(df['team'] > df['opp_slug'], df['team'], df['opp_slug'])
#df['t2'] = np.where(df['team'] > df['opp_slug'], df['opp_slug'], df['team'])
#df = df.drop_duplicates(subset=['t1', 'Date'], keep='first')

df_train = df[df['train_test']=='train']
df_test = df[df['train_test']=='test']
X_train = df_train[XCOLS]
X_test= df_test[XCOLS]
y_train = df_train[LABEL_COL]
y_test = df_test[LABEL_COL]

df.to_csv('Documents/bracket-bot/data/train_test_files/unified_train_test.csv', index=False)
X_train.to_csv('Documents/bracket-bot/data/train_test_files/X_train.csv', index=False)
X_test.to_csv('Documents/bracket-bot/data/train_test_files/X_test.csv', index=False)
y_train.to_csv('Documents/bracket-bot/data/train_test_files/y_train.csv', index=False)
y_test.to_csv('Documents/bracket-bot/data/train_test_files/y_test.csv', index=False)
