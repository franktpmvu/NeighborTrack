import cv2
import torch
import vot
import sys
import time
import os
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker

from NeighborTrack.neighbortrack import neighbortrack
from NeighborTrack.NTutils.utils import xy_wh_2_rect

from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *




class OSTRACK(object):
    def __init__(self, tracker,invtracker):
        self.tracker = tracker
        self.invtracker = invtracker
        '''create tracker'''
        '''Alpha-Refine'''
        refine_model_name = 'ARnet_seg_mask_ep0040.pth.tar'
        
        NeighborTrack/trackers/ostrack/pytracking/networks
        project_path = '/data/NeighborTrack/trackers/ostrack/'
        refine_root = os.path.join(project_path, '/data/NeighborTrack/trackers/ostrack/pytracking/networks/')
        #project_path = '/data/MixFormer/external/AR/'
        #refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        print(refine_path)
        self.alpha = ARcm_seg(refine_path, input_sz=384)
        threshold=0.6
        self.THRES = threshold


    def initialize(self, image, mask):
        region = rect_from_mask(mask)

        params = self.tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.tracker_name
        params.param_name = self.tracker_param
        self.tracker = self.tracker.create_tracker(params)
        self.invtracker = self.invtracker.create_tracker(params)
        
        def _build_init_info(box):
            return {'init_bbox': box}

        
        if region is not None:
            assert isinstance(region, (list, tuple))
            assert len(region) == 4, "valid box's foramt is [x,y,w,h]"
            self.tracker.initialize(image, _build_init_info(region))
            self.ntracker = neighbortrack(self.tracker,image,region[:2],region[2:],invtracker=self.invtracker)
            gt_bbox_np = np.array(region).astype(np.float32)
            self.alpha.initialize(image, np.array(gt_bbox_np))


    def track(self, img_RGB):

        '''TRACK'''
        '''base tracker'''
        state = self.ntracker._neighbor_track(img_RGB)
        #outputs = self.tracker.track(img_RGB)
        #pred_bbox = outputs['target_bbox']
        location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
        #x=location[0]
        #y=location[1]
        #w=location[2]
        #h=location[3]

        #print(pred_bbox)
        #x,y,w,h = pred_bbox
        #return vot.Rectangle(x, y, w, h), state['score']
        '''Step2: Mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(location), vis=True)
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        return final_mask, 1

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)



model_name = 'vitb_384_mae_ce_32x4_ep300_neighbor'



#params = vot_params.parameters("baseline", model="mixformer_online_22k.pth.tar")
# params = vot_params.parameters("baseline")


OStrack = Tracker('ostrack', model_name, "vot")
invOStrack = Tracker('ostrack', model_name, "vot")
tracker = OSTRACK(OStrack,invOStrack)
tracker.tracker_name='ostrack'
tracker.tracker_param=model_name

#mixformer = MixFormerOnline(params, "VOT20")
#tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
#handle = vot.VOT("rectangle")
handle = vot.VOT("mask")
selection = handle.region()
imagefile = handle.frame()

if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
mask = make_full_size(selection, (image.shape[1], image.shape[0]))


#tracker.H = image.shape[0]
#tracker.W = image.shape[1]

tracker.initialize(image, mask)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
