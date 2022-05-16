import pandas as pd
import numpy as np
from langdetect import detect #Language detection 
from statsmodels.stats.proportion import proportion_confint

def preproc(data):
    data["language"] = data["question1"].apply(detect)
    data = data.loc[data["language"] == "en", :]
    del data["language"]

    not_ans = data.isna().sum(1) >= 1

    data.loc[:,'all_quest_answ'] = 1

    data.loc[not_ans.index, 'all_quest_answ'] = 0

    data.loc[:, "label"] = 0
    org = data["labels"] == "Review source: Organic"
    data.loc[org, "label"] = 1
    data.loc[~org, "label"] = 0

    for col in [x for x in data.columns if x.startswith("qu")]:
        data[col] = data[col].fillna(value=" ")
        data[col] = data[col].astype(str)

        data[col] = data[col].map(
            lambda x: x.replace("Review collected by and hosted on G2.com.", "")
        )

    questions = data[[x for x in data.columns if x.startswith("qu")]]

    data["review_text"] = questions.agg(" ".join, axis=1)
    return data


def permutation(stat, group1, group2, niters=10000, random_state=None, dist=False):
    """Pre: group 1 and group 2 are arrays that represents the samples,
         niters is the amount of iterations in the permutation algorithm and
         random_state fix the seed of the generator of random numbers.
         stat is the statistic used for the test (for example difference of means).
    Post: If dist= True, returns a tuple with the array of values with the t-statistic,
          the p-value, and de confidence interval for p-value;
          if dist= False returns  p-value, and the confidence interval for p-value.
    """
    np.random.seed()
    g1_n = len(group1)
    fake_mds = np.zeros(niters)
    pooled = np.concatenate([group1, group2])
    for i in np.arange(niters):
        shuffled = np.random.permutation(pooled)
        fake_mds[i] = np.mean(shuffled[:g1_n]) - np.mean(
            shuffled[g1_n:]
        )  # statistic array
    count = np.count_nonzero(np.abs(fake_mds) >= np.abs(stat))
    p = (count + 1) / (niters + 1)  # p-value
    ci_low, ci_upp = proportion_confint(
        count=count, nobs=niters, method="wilson"
    )  # confidence interval
    if dist == True:
        return fake_mds, p, ci_low, ci_upp
    return p, ci_low, ci_upp
