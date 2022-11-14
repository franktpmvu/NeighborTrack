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

#from lib.test.tracker.mixformer_online import MixFormerOnline
#from pytracking.ARcm_seg import ARcm_seg
#from pytracking.vot20_utils import *

#import lib.test.parameter.mixformer_online as vot_params



class OSTRACK(object):
    def __init__(self, tracker,invtracker):
        self.tracker = tracker
        self.invtracker = invtracker
        '''create tracker'''
        '''Alpha-Refine'''
        #project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        #refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        #refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        #print(refine_path)
        #self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image, region):
        
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
        self.ntracker.ls_add_mode=0


    def track(self, img_RGB):

        '''TRACK'''
        '''base tracker'''
        state = self.ntracker._neighbor_track(img_RGB)
        #outputs = self.tracker.track(img_RGB)
        #pred_bbox = outputs['target_bbox']
        location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
        x=location[0]
        y=location[1]
        w=location[2]
        h=location[3]

        #print(pred_bbox)
        #x,y,w,h = pred_bbox
        return vot.Rectangle(x, y, w, h), state['score']
        '''Step2: Mask report'''
        #pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        #final_mask = (pred_mask > self.THRES).astype(np.uint8)
        #return final_mask, 1




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
handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()

if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
#mask = make_full_size(selection, (image.shape[1], image.shape[0]))


#tracker.H = image.shape[0]
#tracker.W = image.shape[1]

tracker.initialize(image, selection)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
