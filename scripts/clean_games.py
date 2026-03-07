import pandas as pd
import numpy as np
import itertools
import json
import warnings
import re


warnings.filterwarnings("ignore")

YEARS = np.arange(2021, 2027, 1)

with open("Documents/bracket-bot/data/schools.json", "r") as file:
    SCHOOLS = json.load(file)

SCHOOLS = list(SCHOOLS['schools'])


def clean_game_df(df):
    col_prefixes = []
    for i in df.columns:
        if i.startswith('Opponent'):
            col_prefixes.append('opp_')
        elif i.startswith('Score'):
            col_prefixes.append('score_')
        elif i.startswith('Defensive'):
            col_prefixes.append('opp_')
        else:
            col_prefixes.append('')

    cols = list(df.iloc[0,:])
    df.columns = [col_prefixes[x] + cols[x] for x in range(0,len(cols))]
    df.columns = ['site' if x.startswith('Unnamed') else x for x in df.columns]
    df = df[(df['Rk'] != 'Rk') & (~df['Rk'].isna())]
    del df['Rk']
    df = df.set_index('Gtm')

    return df

def window_features(df, periods=[3, 6, None]):
    #columns = ['score_Tm', 'score_Opp', 'FG', 'FGA', '3P', '3PA', '3P%','2P',	'2PA',	'2P%',	'eFG%',	'FT',	'FTA',	'FT%',	'ORB',	'DRB',	'TRB',	'AST',	'STL',	'BLK',	'TOV',	'PF',	'opp_FG',	'opp_FGA',	'opp_FG%',	'opp_3P',	'opp_3PA',	'opp_3P%',	'opp_2P',	'opp_2PA',	'opp_2P%',	'opp_eFG%',	'opp_FT',	'opp_FTA',	'opp_FT%',	'opp_ORB',	'opp_DRB',	'opp_TRB',	'opp_AST',	'opp_STL',	'opp_BLK',	'opp_TOV',	'opp_PF',	'ORtg',	'DRtg',	'Pace',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'TOV%',	'ORB%',	'FT/FGA',	'opp_TOV%',	'opp_ORB%',	'opp_FT/FGA'
    #    ]
    columns = ['score_Rslt','score_Tm',	'score_Opp','FG','FGA','FG%','3P','3PA','3P%','2P',	'2PA',	'2P%',	'eFG%',	'FT','FTA',	'FT%','ORB','DRB',	'TRB',	'AST',	'STL',	'BLK',	'TOV',	'PF',	'opp_FG',	'opp_FGA',	'opp_FG%',	'opp_3P',	'opp_3PA',	'opp_3P%',	'opp_2P',	'opp_2PA',	'opp_2P%',	'opp_eFG%',	'opp_FT',	'opp_FTA',	'opp_FT%',	'opp_ORB',	'opp_DRB',	'opp_TRB',	'opp_AST',	'opp_STL',	'opp_BLK',	'opp_TOV',	'opp_PF',	'ORtg',	'DRtg',	'Pace',	'FTr',	'3PAr',	'TS%',	'TRB%',	'AST%',	'STL%',	'BLK%',	'TOV%',	'ORB%',	'FT/FGA',	'opp_TOV%',	'opp_ORB%',	'opp_FT/FGA',	'point_diff',	'3P_pt_share',	'opp_3P_pt_share',	'2P%_3P%_ratio',	'opp_2P%_3P%_ratio',	'ORB_share',	'opp_ORB_share',	'FG_FT_ratio',	'opp_FG_FT_ratio',	'FG_TRB_ratio',	'opp_FG_TRB_ratio',	'FG_AST_ratio',	'opp_FG_AST_ratio',	'FG_STL_ratio',	'opp_FG_STL_ratio',	'FG_BLK_ratio',	'opp_FG_BLK_ratio',	'FG_TOV_ratio',	'opp_FG_TOV_ratio',	'FG_PF_ratio',	'opp_FG_PF_ratio',	'FT_TRB_ratio',	'opp_FT_TRB_ratio',	'FT_AST_ratio',	'opp_FT_AST_ratio',	'FT_STL_ratio',	'opp_FT_STL_ratio',	'FT_BLK_ratio',	'opp_FT_BLK_ratio',	'FT_TOV_ratio',	'opp_FT_TOV_ratio',	'FT_PF_ratio',	'opp_FT_PF_ratio',	'TRB_AST_ratio',	'opp_TRB_AST_ratio',	'TRB_STL_ratio',	'opp_TRB_STL_ratio',	'TRB_BLK_ratio',	'opp_TRB_BLK_ratio',	'TRB_TOV_ratio',	'opp_TRB_TOV_ratio',	'TRB_PF_ratio',	'opp_TRB_PF_ratio',	'AST_STL_ratio',	'opp_AST_STL_ratio',	'AST_BLK_ratio',	'opp_AST_BLK_ratio',	'AST_TOV_ratio',	'opp_AST_TOV_ratio',	'AST_PF_ratio',	'opp_AST_PF_ratio',	'STL_BLK_ratio',	'opp_STL_BLK_ratio',	'STL_TOV_ratio',	'opp_STL_TOV_ratio',	'STL_PF_ratio',	'opp_STL_PF_ratio',	'BLK_TOV_ratio',	'opp_BLK_TOV_ratio',	'BLK_PF_ratio',	'opp_BLK_PF_ratio',	'TOV_PF_ratio',	'opp_TOV_PF_ratio',	'team_opp_opp_FG_ratio',	'team_opp_opp_FGA_ratio',	'team_opp_opp_FG%_ratio',	'team_opp_opp_3P_ratio',	'team_opp_opp_3PA_ratio',	'team_opp_opp_3P%_ratio',	'team_opp_opp_2P_ratio',	'team_opp_opp_2PA_ratio',	'team_opp_opp_2P%_ratio',	'team_opp_opp_eFG%_ratio',	'team_opp_opp_FT_ratio',	'team_opp_opp_FTA_ratio',	'team_opp_opp_FT%_ratio',	'team_opp_opp_ORB_ratio',	'team_opp_opp_DRB_ratio',	'team_opp_opp_TRB_ratio',	'team_opp_opp_AST_ratio',	'team_opp_opp_STL_ratio',	'team_opp_opp_BLK_ratio',	'team_opp_opp_TOV_ratio',	'team_opp_opp_PF_ratio',	'team_opp_opp_TOV%_ratio',	'team_opp_opp_ORB%_ratio',	'team_opp_opp_FT/FGA_ratio',	'team_opp_opp_3P_pt_share_ratio',	'team_opp_opp_2P%_3P%_ratio_ratio',	'team_opp_opp_ORB_share_ratio',
        ]

    for period in periods:
        for column in columns:
            if period:
                df[f'{column}_l{period}_sum'] = df[column].rolling(period).sum().shift(1)
                df[f'{column}_l{period}_min'] = df[column].rolling(period).min().shift(1)
                df[f'{column}_l{period}_max'] = df[column].rolling(period).max().shift(1)
                df[f'{column}_l{period}_median'] = df[column].rolling(period).median().shift(1)

            else:
                df[f'{column}_cum_sum'] = df[column].cumsum().shift(1)
                df[f'{column}_cum_min'] = df[column].cummin().shift(1)
                df[f'{column}_cum_max'] = df[column].cummax().shift(1)
                df[f'{column}_cum_median'] = df[column].expanding().median().shift(1)
                # .expanding().median().shift(6)

    return df


def transform_features(df):
    df['site'] = df['site'].fillna('H')
    df['Type'] = np.where(~df['Type'].isin(['REG (Conf)', 'REG (Non-Conf)', 'CTOURN']), 'PT', df['Type'])
    df['score_Rslt'] = np.where(df['score_Rslt']=='W', 1, 0)
    df['point_diff'] = df['score_Tm'] - df['score_Opp']
    df['3P_pt_share'] = df['3P'] * 3 / df['score_Tm']
    df['opp_3P_pt_share'] = df['opp_3P'] * 3 / df['score_Opp']
    df['2P%_3P%_ratio'] = df['2P%'] / df['3P%']
    df['opp_2P%_3P%_ratio'] = df['opp_2P%'] / df['opp_3P%']
    df['ORB_share'] = df['ORB'] / df['TRB']
    df['opp_ORB_share'] = df['opp_ORB'] / df['opp_TRB']
    
    opp_cols = [x for x in df.columns if x.startswith('opp_')] # not including ratios of ratios so compute this first
    ratio_cols = ['FG', 'FT', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    ratio_pairs = list(itertools.combinations(ratio_cols,2))
    for k in ratio_pairs:
        col_name = f'{k[0]}_{k[1]}_ratio'
        opp_col_name = 'opp_'+col_name
        df[col_name] = df[k[0]] / df[k[1]]
        df[opp_col_name] = df['opp_'+k[0]] / df['opp_'+k[1]]

    for k in opp_cols:
        col_name = f'team_opp_{k}_ratio'
        df[col_name] = df[k[4:]] / df[k]

    return df


for year in YEARS:
    df_year = pd.DataFrame()
    for school in SCHOOLS:
        try:
            df_basic = pd.read_csv(f'Documents/bracket-bot/data/games/{school}_{str(year)}_basic.csv')
            df_advanced = pd.read_csv(f'Documents/bracket-bot/data/games/{school}_{str(year)}_advanced.csv')
        except:
            print(f'no games for {school} in {year}')
            continue
        df_basic = clean_game_df(df_basic)
        df_advanced = clean_game_df(df_advanced)
        df = df_basic.join(df_advanced, rsuffix='_drop')
        df = df.filter(regex='^(?!.*_drop$)')
        for i in df.columns[5:]:
            try:
                df[i] = pd.to_numeric(df[i])
            except:
                pass
        df = transform_features(df)
        df = window_features(df)
        df = df.reset_index()
        df['team'] = school
        df['opp_slug'] = (
                df['Opp']
                .str.lower()
                .str.replace('@', ' ', regex=False)
                .str.replace(r"[^\w\s-]", "", regex=True)
                .str.replace(r"\s+", "-", regex=True)
                .str.replace(r"-+", "-", regex=True)
                .str.strip("-")
            )
        
        df_year = pd.concat([df_year, df], axis=0, ignore_index=True)
        print(f'{year} {school} complete')
    df_year.to_csv(f'Documents/bracket-bot/data/years/games_{year}.csv', index=False)
    print(f'{year} complete')
    print('\n\n')

    
        

