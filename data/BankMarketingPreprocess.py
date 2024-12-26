import pandas as pd
import numpy as np

df = pd.read_csv('./bank/bank-full-shuffle.csv', sep=';')

onehotCols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome' ]


def toOneHot(item, coldict):
    # return np.array([coldict[item]])
    vec = np.zeros(len(coldict), dtype=int)
    vec[coldict[item]] = 1
    return vec


data = []
for col in df.columns:
    if col in onehotCols:
        colset = set(df[col])
        coldict = dict()
        for i, tp in enumerate(sorted(colset)): coldict[tp] = i
        d = np.array(df[col].apply(lambda item:toOneHot(item, coldict)).tolist())
        data.append(d)
    else:
        print(col)
        d = df[col].values.reshape(-1,1)
        if col != 'y':
            dmean = d.mean()
            dstd = d.std()
            d = (d-dmean)/dstd
        data.append(d)

ans = np.concatenate(data, axis=1)
ansdf = pd.DataFrame(ans)
idx = list(range(ansdf.shape[0]))
ansdf = ansdf.iloc[idx]
traindf = ansdf.iloc[:int(ansdf.shape[0]*0.8)]
testdf = ansdf.iloc[int(ansdf.shape[0]*0.8):]

traindf.to_csv('./1/train.csv', index=None, header=None)
testdf.to_csv('./1/test.csv', index=None, header=None)

