import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging

import numpy as np


matplotlib.rcParams.update({'font.size': 38, 'errorbar.capsize': 12})
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

N_RAND_LABELS = [
    2,
    4,
    8,
    16,
    32,
    64
]

N = 6
FIG_SIZE = (12, 9)
iso_ll_means = [4.198138894, 6.574704497, 11.96620541, 10.05824439, 12.70133378, 12.75857178]
iso_ll_std = [4.150568295, 4.066506494, 7.371413984, 2.979296132, 0.1717140014, 4.59E-14]

pwl_ll_means = [3.338810787, 5.980470521, 11.85872784, 9.981002762, 12.61258184, 12.66935761]
pwl_ll_std = [3.344179849, 3.589407203, 7.442052843, 2.993812608, 0.1703272999, 3.91E-14]

histo_ll_means = [15.04451423, 16.13371734, 16.81263092, 17.0046108, 16.99680663, 16.99095745]
histo_ll_std = [0.5577271915, 0.4824096804, 0.3036567293, 0.2635683877, 0.01754756479, 3.30E-14]

ind = np.arange(N)  # the x locations for the groups
width = 0.32       # the width of the bars

bottom_line = -5
fig, ax = plt.subplots(figsize=FIG_SIZE)
rects1 = ax.bar(ind, histo_ll_means, width, color='#FA7800', yerr=histo_ll_std, )
# rects2 = ax.bar(ind + width, pwl_ll_means, width, color='#21C2CF',
#                 yerr=pwl_ll_std, )
# rects3 = ax.bar(ind + width * 2, iso_ll_means, width,
#                 color='#72327a', yerr=iso_ll_std, )
rects2 = ax.bar(ind + width, iso_ll_means, width, color='#21C2CF',
                yerr=iso_ll_std, )

ax.plot([-0.5, 6], [0, 0], "k--", linewidth=3)
fmt = '%+.0f%%'  # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)

# add some text for labels, title and axes ticks
ax.set_ylabel('marginal log-likelihood\nrelative improvement')
ax.set_xlabel('semantic code length')
# ax.set_title('Scores by # privileged features and leaf type')
ax.set_xticks(ind + (width * 2) / 2)
ax.set_xticklabels(N_RAND_LABELS)
ax.set_ylim([-3, ax.get_ylim()[-1]])


# leg = ax.legend((rects1[0], rects2[0], rects3[0]), ('histo', 'pwl', 'iso'), loc=4)
leg = ax.legend((rects1[0], rects2[0]), ('hist', 'iso'), loc=4)
leg.get_frame().set_alpha(1.0)
plt.tight_layout()

# def autolabel(rects):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

output = '/home/emanuele/Desktop/marg-test-ll'
pp = PdfPages(output + '.pdf')
pp.savefig(fig)
pp.close()
logging.info('Saved image to pdf {}'.format(output))

# plt.show()


histo_mpe_acc_means = [2.36909111, 3.62559504, 4.513450681, 4.679508469, 4.624155873, 4.626369977]
histo_mpe_acc_std = [0.4967444481, 0.2164987147, 0.1680423134,
                     0.07953072094, 0.009963467287, 0.003321155762]

pwl_mpe_acc_means = [1.925277684, 3.313138113, 2.127229889, 4.437338719, 4.827779648, 4.835633345]
pwl_mpe_acc_std = [0.7737792955, 0.5851636092, 2.10603067, 0.7095034483, 0.02356109054, 0]

iso_mpe_acc_means = [1.93718452, 3.35950645, 2.115535614, 4.443073472, 4.831183399, 4.843522154]
iso_mpe_acc_std = [0.7786935514, 0.5957563073, 2.11560275,
                   0.7110486084, 0.04374649467, 0.004486819966]

ind = np.arange(N)  # the x locations for the groups
width = 0.32       # the width of the bars


fig, ax = plt.subplots(figsize=FIG_SIZE)

bottom_line = -1
rects1 = ax.bar(ind, histo_mpe_acc_means, width, color='#FA7800',
                yerr=histo_mpe_acc_std)
# rects2 = ax.bar(ind + width, pwl_mpe_acc_means, width, color='#21C2CF',
#                 #yerr=(np.minimum(0, -np.array(pwl_mpe_acc_std)), pwl_mpe_acc_std)
#                 yerr=(pwl_mpe_acc_std))
# rects3 = ax.bar(ind + width * 2, iso_mpe_acc_means, width,
#                 color='#72327a', yerr=iso_mpe_acc_std)
rects2 = ax.bar(ind + width, iso_mpe_acc_means, width, color='#21C2CF',
                #yerr=(np.minimum(0, -np.array(pwl_mpe_acc_std)), pwl_mpe_acc_std)
                yerr=(iso_mpe_acc_std))

fmt = '%+.0f%%'  # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
ax.plot([-0.5, 6], [0, 0], "k--", linewidth=3)

# add some text for labels, title and axes ticks
ax.set_ylabel('accuracy\nrelative improvement')
ax.set_xlabel('semantic code length')
# ax.set_title('Scores by # privileged features and leaf type')
ax.set_xticks(ind + (width * 2) / 2)
ax.set_xticklabels(N_RAND_LABELS)
ax.set_ylim([-1, ax.get_ylim()[-1]])

# leg = ax.legend((rects1[0], rects2[0], rects3[0]), ('histo', 'pwl', 'iso'), loc=4)
leg = ax.legend((rects1[0], rects2[0]), ('hist', 'iso'), loc=4)
leg.get_frame().set_alpha(1.0)
plt.tight_layout()

# def autolabel(rects):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

output = '/home/emanuele/Desktop/test-mpe-acc'
pp = PdfPages(output + '.pdf')
pp.savefig(fig)
pp.close()
logging.info('Saved image to pdf {}'.format(output))
