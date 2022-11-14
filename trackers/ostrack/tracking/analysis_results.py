import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""

#-------------lasot---------------
dataset_name = 'lasot'
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack384'))
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_neighbor', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack384_neighbor'))
dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

#-----------------got10k-----------------
'''
dataset_name = 'got10k_val'
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_ori/got10k', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_ori'))
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_neighbor_n9/got10k', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_neighbor_n9'))
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_neighbor_n18/got10k', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_neighbor_n18'))
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_neighbor_n27/got10k', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_neighbor_n27'))
dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
'''
#--------------trackingnet---------------
'''
dataset_name = 'trackingnet'
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_ori/trackingnet', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_ori'))
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_neighbor_n9/trackingnet', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_neighbor_n9'))
trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_neighbor_n18/trackingnet', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack_neighbor_n18'))



dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
'''