import pandas as pd
import numpy as np
import datetime
from itertools import product
from scipy import interpolate

def interpolateFun0(x):
    global huh
    x = x.reset_index(drop=True)
    g = x['outcome']

    if g.shape[0] < 3:
        x['filled'] = g
        x['outcome'] = x['filled']
        return x

    out = g.values.copy()
    value_locs = np.where(~g.isnull())[0]

    if len(value_locs) == 0:
        x['filled'] = np.full_like(out, .05)
        x['outcome'] = x['filled']
        return x

    if len(value_locs) == 1:
        g[g.isnull()] = .95 if g[value_locs[0]] else .05

        x['filled'] = g
        x['outcome'] = x['filled']

        return x

    if value_locs[0]:
        end = value_locs[0]
        fillval = .95 if g[end] else .05
        out[0:end] = np.full(end, fillval)

    for i in range(0, len(value_locs) - 1):
        beg = value_locs[i]
        end = value_locs[i + 1]

        out[beg+1:end] = np.interp(range(beg+1, end), [beg, end], [g[beg], g[end]])

    if end < (len(g) - 1):
        beg = value_locs[-1]
        end = len(out)

        fillval = .95 if g[beg] else .05
        out[beg+1:] = np.full(end-beg-1, fillval)

    x['filled'] = out
    x['outcome'] = x['filled']

    return x

def interpolateFun1(x):
    g = x['outcome']
    missing_index = g.isnull()
    border_fill = 0.1
    if g.index[0] in missing_index:
        g[g.index[0]] = border_fill
    if g.index[-1] in missing_index:
        g[g.index[-1]] = border_fill
    known_index = ~g.isnull()
    try:
        f = interpolate.interp1d(g[known_index].index, g[known_index], kind='linear')
        x['filled'] = [f(x) for x in g.index]
        x['filled'] = np.interp(g.index, g[known_index].index, g[known_index])
    except ValueError:
        x['filled'] = x['outcome']
    return x

if __name__ == '__main__':
    ppl = pd.read_csv('../input/people.csv')


    p_logi = ppl.select_dtypes(include=['bool']).columns
    ppl[p_logi] = ppl[p_logi].astype('int')
    del p_logi

    ppl['date'] = pd.to_datetime(ppl['date'])

    activs = pd.read_csv('../input/act_train.csv')
    TestActivs = pd.read_csv('../input/act_test.csv')
    TestActivs['outcome'] = np.nan
    activs = pd.concat([activs, TestActivs], axis=0)
    del TestActivs
    activs = activs[['people_id', 'outcome', 'activity_id', 'date']]
    d1 = pd.merge(activs, ppl, on='people_id', how='right')

    testset = ppl[ppl['people_id'].isin(d1[d1['outcome'].isnull()]['people_id'])].index

    d1['activdate'] = pd.to_datetime(d1['date_x'])

    del activs
    minactivdate = min(d1['activdate'])
    maxactivdate = max(d1['activdate'])
    alldays = [maxactivdate - datetime.timedelta(days=x) for x in range(0, (maxactivdate - minactivdate).days+1)][::-1]

    grid_left = set(d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])]['group_1'])

    allCompaniesAndDays = pd.DataFrame.from_records(product(grid_left, alldays))

    allCompaniesAndDays.columns = ['group_1', 'date_p']

    allCompaniesAndDays.sort_values(['group_1', 'date_p'], inplace=True)

    meanbycomdate = d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])].groupby(['group_1', 'activdate'])['outcome'].agg('mean')

    meanbycomdate = meanbycomdate.to_frame().reset_index()
    allCompaniesAndDays = pd.merge(allCompaniesAndDays, meanbycomdate, left_on=['group_1', 'date_p'], right_on=['group_1', 'activdate'], how='left')
    allCompaniesAndDays.drop('activdate', axis=1, inplace=True)
    allCompaniesAndDays.sort_values(['group_1', 'date_p'], inplace=True)

    allCompaniesAndDays = allCompaniesAndDays.groupby('group_1').apply(interpolateFun0)

    d1 = pd.merge(d1, allCompaniesAndDays,
                  left_on=['group_1', 'activdate'], right_on=['group_1', 'date_p'], how='left')

    testsetdt = d1[d1['people_id'].isin(ppl.iloc[testset]['people_id'])][['activity_id', 'filled']]

    testsetdt.columns = [testsetdt.columns[0], 'outcome']
    testsetdt.to_csv('input/Submissionaid.csv', index=False)