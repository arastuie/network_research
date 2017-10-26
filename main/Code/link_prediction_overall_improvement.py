import pickle
import numpy as np

path = '/shared/Results/EgocentricLinkPrediction/main/lp/fb'
top_k_values = [10, 15, 20, 25, 30]

with open('{0}/pickle-files/lower-6/temp/total-result-percent-imp.pckl'.format(path), 'rb') as f:
    lower_percent_imp_aa, lower_percent_imp_cn = pickle.load(f)

lower_imp_aa = []
lower_imp_cn = []
for k in top_k_values:
    lower_imp_aa += lower_percent_imp_aa[k]
    lower_imp_cn += lower_percent_imp_cn[k]


with open('{0}/pickle-files/after-6/temp/total-result-percent-imp.pckl'.format(path), 'rb') as f:
    after_percent_imp_aa, after_percent_imp_cn = pickle.load(f)

after_imp_aa = []
after_imp_cn = []
for k in top_k_values:
    after_imp_aa += after_percent_imp_aa[k]
    after_imp_cn += after_percent_imp_cn[k]


aa = lower_imp_aa + after_imp_aa
cn = lower_imp_cn + after_imp_cn

print("Overall improvement CN Before PYMK: {0:.4f}%".format(np.mean(lower_imp_cn) * 100))
print("Overall improvement CN After PYMK: {0:.4f}%".format(np.mean(after_imp_cn) * 100))
print("Overall improvement AA Before PYMK: {0:.4f}%".format(np.mean(lower_imp_aa) * 100))
print("Overall improvement AA After PYMK: {0:.4f}%".format(np.mean(after_imp_aa) * 100))
print("Overall improvement CN: {0:.4f}%".format(np.mean(cn) * 100))
print("Overall improvement AA: {0:.4f}%".format(np.mean(aa) * 100))
