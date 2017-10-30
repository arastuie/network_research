import pickle
import numpy as np
import helpers as h
import matplotlib.pyplot as plt

path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb'

paths = {
    'Global': ['global/lower-6', 'global/after-6'],
    'Local': ['local/lower-6', 'local/after-6']
}

names = ['lower-6', 'after-6']
for ps in paths:
    for i in range(len(paths[ps])):
        with open('{0}/pickle-files/{1}/temp/total-result.pckl'.format(path, paths[ps][i]), 'rb') as f:
            mfems, mnfems = pickle.load(f)

        plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/local/lower-6'

        plt.rc('legend', fontsize=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        # plt.step(np.sort(mfems), np.arange(1, len(mfems) + 1) / np.float(len(mfems)), alpha=0.9, color='r',
        #          label='Formed Edges: {0:.4f}'.format(np.mean(mfems)), lw=3)

        h.add_ecdf_with_band_plot_undirected(mfems, 'Formed', 'r')

        # plt.step(np.sort(mnfems), np.arange(1, len(mfems) + 1) / np.float(len(mfems)), alpha=0.9, color='b',
        #          label='Not Formed Edges: {0:.4f}'.format(np.mean(mnfems)), lw=3)

        h.add_ecdf_with_band_plot_undirected(mnfems, 'Not Formed', 'b')

        plt.ylabel('Empirical CDF', fontsize=20)
        plt.xlabel('Mean Normalized {0} Degree'.format(ps), fontsize=20)
        plt.legend(loc='lower right')
        plt.tight_layout()
        current_fig = plt.gcf()
        current_fig.savefig('{0}/cdf/{1}-{2}-overall-mean-normal.pdf'.format(path, ps, names[i]), format='pdf')
        # current_fig.savefig('{0}/cdf/{1}-{2}-overall-mean-normal-png.png'.format(path, ps, names[i]))
        plt.clf()
        print("Number of egonets analyzed for {0} {1}: {2}".format(ps, names[i], len(mfems)))

print("Done")


## On top of eachother

# print("Analysing ego centric networks...")
# path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/'
#
# paths = {
#     'Global': ['global/lower-6', 'global/after-6'],
#     'Local': ['local/lower-6', 'local/after-6']
# }
#
# names = ['lower-6', 'after-6']
# for ps in paths:
#     with open('{0}/{1}/temp/total-result.pckl'.format(path, paths[ps][0]), 'rb') as f:
#         lmfems, lmnfems = pickle.load(f)
#
#     with open('{0}/{1}/temp/total-result.pckl'.format(path, paths[ps][1]), 'rb') as f:
#         amfems, amnfems = pickle.load(f)
#
#     plot_path = '/shared/Results/EgocentricLinkPrediction/main/empirical/fb/pickle-files/local/lower-6'
#
#     plt.hist(lmfems, 1000, normed=True, cumulative=True, histtype='step', alpha=0.9, color='r',
#              label='Before PYMK Formed Edges: {0:.4f}'.format(np.mean(lmfems)))
#
#     plt.hist(lmnfems, 1000, normed=True, cumulative=True, histtype='step', alpha=0.9, color='b',
#              label='Before PYMK Not Formed Edges: {0:.4f}'.format(np.mean(lmnfems)))
#
#     plt.hist(amfems, 1000, normed=True, cumulative=True, histtype='step', alpha=0.9, color='y',
#              label='After PYMK Formed Edges: {0:.4f}'.format(np.mean(amfems)))
#
#     plt.hist(amnfems, 1000, normed=True, cumulative=True, histtype='step', alpha=0.9, color='g',
#              label='After PYMK Not Formed Edges: {0:.4f}'.format(np.mean(amnfems)))
#
#     # plt.hist(mfems, color='r', alpha=0.8, weights=np.zeros_like(mfems) + 1. / len(mfems),
#     #          label="Formed Edges: {0:.2f}".format(np.mean(mfems)))
#
#     # plt.hist(mnfems, color='b', alpha=0.5, weights=np.zeros_like(mnfems) + 1. / len(mnfems),
#     #          label="Not Formed Edges: {0:.4f}".format(np.mean(mnfems)))
#
#     plt.legend(loc='upper right')
#     plt.ylabel('Relative Frequency')
#     plt.xlabel('Mean Normalized {0} Degree of Common Neighbors'.format(ps))
#
#     current_fig = plt.gcf()
#     current_fig.savefig('{0}/plots/{1}-overall-mean-normal.eps'.format(path, ps), format='eps', dpi=1000)
#     current_fig.savefig('{0}/plots/{1}-overall-mean-normal-png.png'.format(path, ps))
#     plt.clf()
#
# print("Done")