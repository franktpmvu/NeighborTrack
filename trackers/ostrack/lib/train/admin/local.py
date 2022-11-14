class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/OSTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/OSTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/OSTrack/pretrained_networks'
        self.lasot_dir = '/data/OSTrack/data/lasot'
        self.got10k_dir = '/data/OSTrack/data/got10k/train'
        self.got10k_val_dir = '/data/OSTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/data/OSTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/OSTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/data/OSTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/data/OSTrack/data/trackingnet_lmdb'
        self.coco_dir = '/data/OSTrack/data/coco'
        self.coco_lmdb_dir = '/data/OSTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data/OSTrack/data/vid'
        self.imagenet_lmdb_dir = '/data/OSTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
