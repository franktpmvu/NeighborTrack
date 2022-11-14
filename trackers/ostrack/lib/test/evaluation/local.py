from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/TracKit/dataset/got10k_lmdb'
    settings.got10k_path = '/data/TracKit/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/OSTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/data/OSTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/OSTrack/data/lasot_lmdb'
    settings.lasot_path = '/data/OSTrack/data/lasot'
    settings.network_path = '/data/NeighborTrack/trackers/ostrack/models'    # Where tracking networks are stored.
    settings.nfs_path = '/data/OSTrack/data/nfs'
    settings.otb_path = '/data/OSTrack/data/otb'
    settings.prj_dir = '/data/NeighborTrack/trackers/ostrack'
    settings.result_plot_path = '/data/NeighborTrack/trackers/ostrack/output/test/result_plots'
    settings.results_path = '/data/NeighborTrack/trackers/ostrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/NeighborTrack/trackers/ostrack/output'
    settings.segmentation_path = '/data/NeighborTrack/trackers/ostrack/output/test/segmentation_results'
    settings.tc128_path = '/data/OSTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/OSTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/TracKit/dataset/trackingnet'
    settings.uav_path = '/data/OSTrack/data/uav'
    settings.vot18_path = '/data/OSTrack/data/vot2018'
    settings.vot22_path = '/data/OSTrack/data/vot2022'
    settings.vot_path = '/data/OSTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings