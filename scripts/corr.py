import pandas as pd
import numpy as np


corr = pd.read_csv('cormat.csv')
corr.index = corr.columns
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

threshold = 0.995

high_corr = [
    column for column in upper.columns
    if any(upper[column] > threshold)
]

pairs = (
    upper.stack()
         .reset_index()
         .rename(columns={0:'corr'})
)

pairs = pairs[pairs['corr'] > threshold]
print(pairs.sort_values('corr', ascending=False))
pd.DataFrame({'high_corr':pd.Series(high_corr)}).to_csv('Documents/bracket-bot/data/high_corr.csv',index=False)