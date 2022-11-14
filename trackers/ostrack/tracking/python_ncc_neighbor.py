#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy
import collections
from NeighborTrack.neighbortrack import neighbortrack
from NeighborTrack.NTutils.utils import xy_wh_2_rect
print(vot.__file__)
#print(vot.__version__)


class normal_NCCTracker(object):

    def __init__(self, image, region):
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        
    def track(self, image):#original version
        
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]], max_val

    
    
        
    def initialize(self,image,init_info):
        #init_info = {'init_bbox':[x,y,w,h]}
        
        xywh = init_info['init_bbox']
        self.window = max(xywh[2], xywh[3]) * 2

        
        left = max(xywh[0], 0)
        top = max(xywh[1], 0)
        left = min(xywh[0],image.shape[1])
        top = min(xywh[1],image.shape[0])
        
        right = min(xywh[0] + xywh[2], image.shape[1] - 1)
        bottom = min(xywh[1] + xywh[3], image.shape[0] - 1)
        
        self.template = image[int(top):int(bottom), int(left):int(right)]
        #self.template = self.template.remove([])
        #print(self.template[0])
        #print(len(self.template))
        #print(image.shape[1])
        #print(image.shape[0])
        #print(xywh)
        #print([left,top,right,bottom])

        self.position = (xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2)
        self.size = (xywh[2], xywh[3])

    
    def track_neighbor(self,image,th):
        
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0] or self.template.size==0:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]],0,[],[]

        cut = image[int(top):int(bottom), int(left):int(right)]
        #print(len(self.template))
        #print(self.template.size)

        #print(len(cut))
        #print(self.template[0])
        #print(cut[0])
        #if len(self.template)<len(cut)/2 or:
        #    return [],0,[], []

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        

        
        
        neighbors_value = cv2.inRange(matches,max_val*th,max_val)
        neighbors_index = cv2.findNonZero(neighbors_value)
        if not neighbors_index is None:
            neighbors_value_only = [matches[x[0][1],x[0][0]] for x in neighbors_index]
            #self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)
            neighbors_xywh=[[left+x[0][0],top+x[0][1], self.size[0], self.size[1]] for x in neighbors_index]
        else:
            neighbors_value_only=[]
            neighbors_xywh=[]
        
        xywh=[left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]
        print(len(neighbors_xywh))
        return xywh,max_val,neighbors_xywh, neighbors_value_only
    
    def update_center(self,xywh):
        #print(xywh)
        x,y,w,h = xywh
        self.position = (x+float(w)/2,y+float(h)/2)
        
        
        
class NCCTracker(object):

    def __init__(self, image, region):
        self.tracker = normal_NCCTracker(image,region)
        self.invtracker = normal_NCCTracker(image,region)
        
        self.ntracker = neighbortrack(self.tracker,image,region[:2],region[2:],invtracker=self.invtracker)
        self.ntracker.ls_add_mode=0


    def track(self, image):
        
        state = self.ntracker._neighbor_track(image)
        location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
        x,y,w,h = location



        return vot.Rectangle(x, y, w, h), state['score']

    
    

#print(vot.VOT)
#dsa
handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
tracker = NCCTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
    
    