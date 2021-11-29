import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
from statsmodels.discrete.count_model import  ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.stats.multitest import multipletests

#Printing whole dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#Loading the data
ab_data = pd.read_csv("AB_test_Homework_data - InApp_data.csv", index_col=0)

group_names = ab_data['test_group'].unique()
#dummy variables
for i in range(len(group_names)):
    ab_data[group_names[i]] = (ab_data['test_group'] == group_names[i]).astype(int)

#EXPLORING DATA
'''
np.round(ab_data.describe(), 2)

#that's the function I've could used at the interview...
ab_data.iloc[:, 0].value_counts()
ab_data.iloc[:, 1].value_counts()

#Why are there 67 ppl in ref group getting a hint? Should I remove them from experiment?
ab_data[['test_group', 'is_hinted', 'm_clicks_total']].groupby(['test_group', 'is_hinted']).count()

#Comparing differences between reference and other groups
ab_data[['DNA-XYZ-ref', 'm_clicks_total']].groupby('DNA-XYZ-ref').mean()
ab_data[['is_hinted', 'm_clicks_total']].groupby('is_hinted').mean() #ref group without hints click less
#Comparing m_clicks_total means for each group
ab_data[['test_group', 'm_clicks_total']].groupby('test_group').agg({'m_clicks_total': [np.std, np.mean, lambda q: np.quantile(q, 0.95)]}) #overdispersion?

#Missing values
ab_data.notna().sum() / ab_data.shape[0] #20% of fst_hint_day missing -- only in reference group
ab_data[(ab_data['DNA-XYZ-ref']) & (ab_data['is_hinted']==1)]['fst_hint_day'].notna().sum() #those in ref group with hints have fst_hint_day
'''

#PLOTS
#Scatter plots
'''
interesting_var = ['age0', 'm_clicks_total', 'm_used_days', 'days_in_test','m_hints_dismissed', 'fst_hint_day']
ab_data[interesting_var].corr()
pd.plotting.scatter_matrix(ab_data[interesting_var])
plt.show()
'''

#Histograms (trying out some distributions)
'''
fig, ax = plt.subplots(2,2, figsize=(8,8))

ax[0,0].hist(ab_data['days_in_test'], density=True, rwidth=0.9)
ax[0,0].set_xlabel('days_in_test')

ax[0,1].hist(ab_data['fst_hint_day'], density=True, rwidth=0.9)
ax[0,1].set_xlabel('fst_hint_day')
x = np.arange(ab_data['fst_hint_day'].max())
ax[0,1].plot(x, st.poisson.pmf(x, mu=ab_data['fst_hint_day'].mean()))

ax[1,0].hist(ab_data['m_hints_dismissed'], density=True, rwidth=0.9)
ax[1,0].set_xlabel('m_hints_dismissed')
x = np.linspace(0, ab_data['m_hints_dismissed'].max(), num=100)
ax[1,0].plot(x, st.expon.pdf(x, scale=1/ab_data['m_hints_dismissed'].mean()))

m_tot = ab_data['m_clicks_total']
ab_data['m_clicks_total_cut'] = m_tot.apply(func=(lambda x: min(x, m_tot.quantile(0.95))) )
ax[1,1].hist(ab_data.m_clicks_total_cut, density=True, bins=int(ab_data['m_clicks_total_cut'].max()))
ax[1,1].set_xlabel('m_clicks_total_cut')
x = np.arange(ab_data['m_clicks_total_cut'].max())
ax[1,1].plot(x, st.poisson.pmf(x, mu=ab_data.m_clicks_total_cut.mean()))

mu = ab_data.m_clicks_total_cut.mean()
sigma2 = ab_data.m_clicks_total_cut.var()
ax[1,1].plot(x, st.nbinom.pmf(x, n = mu/sigma2, p = mu**2 / (sigma2 - mu)))
#problem with assuming poisson / negative binomial distribution, a lot of clicks at the end

plt.show()
'''

#TESTING
#GLOBAL
#is using m feature more probable when showing hint?
ab_data['m_used'] = (ab_data['m_clicks_total'] > 0).astype(int)
'''
cont_table = pd.crosstab(index=ab_data['m_used'], columns=ab_data['is_hinted'])
cont_table / cont_table.sum(axis=0)
#bernard, because only one margin is fixed
st.barnard_exact(cont_table, alternative='greater', pooled=False)

#is using m feature less probable in other ref group?
cont_table = pd.crosstab(index=ab_data['m_used'], columns=ab_data['DNA-XYZ-ref'])
cont_table / cont_table.sum(axis=0)
st.barnard_exact(cont_table, alternative='less', pooled=False)
'''

#Both tests reveal that being in other than ref group (or having shown hint) have significant difference on using feature m


#MULTIPLE

#ab_data[['test_group', 'm_used', 'm_clicks_total']].groupby('test_group').sum().sort_values(['m_used'], ascending=False)
#hypothesis: big-header-sidebar is the best group

def test_mult_m_used(best_group = 'DNA-XYZ-big-header-sidebar', alpha=0.1):
    other_group_names = [x for x in group_names if x != best_group]
    pvals = np.zeros(len(other_group_names))

    for i in range(len(other_group_names)):
        print(other_group_names[i], "\n")
        data = ab_data[((ab_data[best_group]) | (ab_data[other_group_names[i]])).astype(bool)]
        cont_table = pd.crosstab(index=data['m_used'], columns=data[best_group])
        #print(cont_table / cont_table.sum(axis=0))
        pvals[i] = st.barnard_exact(cont_table, alternative='greater', pooled=False).pvalue
    print("p-values: ", pvals)
    print("Bonferroni correction:",  multipletests(pvals, alpha=alpha, method='bonferroni')[1])
    print("BH correction:", multipletests(pvals, alpha=alpha, method='fdr_bh')[1])

#test_mult_m_used()
'''
Barnard exact tests confirm that big-header-sidebar has greater probability of using m feature compared to ref group.
After correcting p-values using bonferroni and benjamini-hochberg we can't be sure if big-header-sidebar has greater
probability than all other groups (than ref group).
'''

#BREAKING CONTINUITY
#In case of the following tests I'm not sure if continuity is an assumption

m_totals = [[] for x in range(len(group_names))]
for i in range(len(group_names)):
    m_totals[i] =  ab_data[ab_data['test_group']==group_names[i]].m_clicks_total

#Testing global difference in m_clicks_total
#st.kruskal(m_totals[0], m_totals[1], m_totals[2], m_totals[3], m_totals[4])

#test revealed that there is significant difference in means of m_clicks_total for different groups

#Testing multiple hypothesis
def test_mult_m_totals(totals, best_group = 1, alpha=0.1):
    best_total = totals[best_group]
    totals = [totals[i] for i in range(len(totals)) if i != best_group]
    pvals = np.zeros(len(totals))
    for i in range(len(totals)):
        stat, pvals[i] = st.mannwhitneyu(best_total, totals[i], alternative='greater')
    print("p-values: ", pvals)
    print("Bonferroni correction:",  multipletests(pvals, alpha=alpha, method='bonferroni')[1])
    print("BH correction:", multipletests(pvals, alpha=alpha, method='fdr_bh')[1])
#test_mult_m_totals(m_totals)

'''
Similarly to the previous tests, Mann-Whitney tests (after correction) cannot confirm if the big-header-sidebar
has the greatest mean of m_clicks_total.
'''

#REGRESSION
#95% of clicks could be from negative binomial
ab_data_95 = ab_data[ab_data['m_clicks_total']<ab_data['m_clicks_total'].quantile(.95)]
low_clicks = ab_data_95['m_clicks_total']
'''
plt.hist(low_clicks ,  density=True, bins=low_clicks.max()+1)
x = np.arange(low_clicks.min(), low_clicks.max()+1)
mu = low_clicks.mean()
sigma2 = low_clicks.var()
plt.plot(x, st.nbinom.pmf(x, p=mu/sigma2, n=mu**2/(sigma2-mu), loc=low_clicks.min()))
print('mu=', mu, 'sigma2=', sigma2)
plt.show()
'''


#Removing 5% of data (after cutting we will be losing 5% of data with large amounts of clicks)
#filling na's with 0 (no influence on estimator)
'''
ab_data_95 = ab_data_95.fillna(value=0)

no_pred = ['country', 'test_group', 'small_huid', 'm_clicks_total', 'm_used',
           'is_hinted', 'm_click_w1', 'm_click_w2', 'fst_hint_day', 'm_hints_accepted ',
           'm_hints_dismissed',  'm_hints_dont_show', 'age0', 'days_in_test',  'DNA-XYZ-ref']
pred = [x for x in ab_data.columns.to_list() if x not in no_pred]

X = ab_data_95[pred]
Y = low_clicks

#models
poiss = Poisson(Y, X).fit()
poiss_zero = ZeroInflatedPoisson(Y,X).fit(maxiter=100)
nb = NegativeBinomial(endog=Y, exog=X).fit()
nb_zero = ZeroInflatedNegativeBinomialP(Y,X).fit(maxiter=100)

#hierarchical testing
1 - st.chi2.cdf(-2*(poiss.llf - poiss_zero.llf), 1)
1 - st.chi2.cdf(-2*(poiss.llf - nb.llf), 1)
1 - st.chi2.cdf(-2*(poiss.llf - nb_zero.llf), 2)
1 - st.chi2.cdf(-2*(nb.llf - nb_zero.llf), 1)

#every extension of poisson regression model is better, negative binomial is better than zero inflated NB

#non-hierarchical comparison
IC = pd.DataFrame()
IC['aic'] = [mod.aic for mod in [poiss, poiss_zero, nb, nb_zero]]
IC['bic'] = [mod.bic for mod in [poiss, poiss_zero, nb, nb_zero]]
IC.index = ['poiss', 'poiss_zero', 'nb', 'nb_zero']
IC.sort_values(['aic'], ascending=True)

#AIC and BIC confirms that NB is the best model of those 4

#best model -- nb, testing its combinations:
nb1 = NegativeBinomial(Y, X[pred[1:]]).fit()
nb2 = NegativeBinomial(Y, X[np.append(pred[2:], 'm_used_days')]).fit()
nb3 = NegativeBinomial(Y, X[pred[2:]]).fit()

1 - st.chi2.cdf(-2*(nb1.llf - nb.llf), 1)
1 - st.chi2.cdf(-2*(nb2.llf - nb.llf), 1)
1 - st.chi2.cdf(-2*(nb3.llf - nb.llf), 2)

#full NB model is better than reduced ones

IC = pd.DataFrame()
IC['aic'] = [mod.aic for mod in [nb, nb1, nb2, nb3]]
IC['bic'] = [mod.bic for mod in [nb, nb1, nb2, nb3]]
IC.index = ['nb', 'nb1', 'nb2', 'nb3']
IC.sort_values(['aic'], ascending=True)

#AIC and BIC confirms previous conclusion

print(nb.summary())
print(nb.conf_int())
'''
'''
Unfortunately, I couldn't find any better tests to test the difference in means (of discrete data) in populations, 
so I had to improvise with Barnard, Mann-Whitney and multiple testing procedures. 
Using them I couldn't asses what group is the best one, so I've tried a few models for discrete data (extensions of Poisson regression).
Based on the best model I could infer that DNA-XYZ-big-header-sidebar may have the biggest influence on m_clicks_total
(the lowest negative influence).
The models were build on 95% of the data, the highest amounts of clicks were removed (were they bots?)
I'm not glad with those conclusions, but I do not have any more time to work on this project.

The other ideas that I've had: 
    - m_clicks_totall looked like mixture of 2 distributions: negative binomial and something like geometric, paretto, logistic?
    Parameters of this mixture could be obtained via EM algorithm. Later it would be possible to do simulations 
    and use Crude Monte Carlo estimators to compare difference in means?
    - some combination of variables could be easier to test (for example, I've tried m_clicks_total / days_in_test)
'''