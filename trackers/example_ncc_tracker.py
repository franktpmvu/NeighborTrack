# example of ncc tracker , func(__init__) and func(track) are original ncc tracker on votchallenge github code, 
# add 3 function to use our method, func(initialize), func(track_neighbor), func(update_center)

# func(initialize)
#  initialize the tracking method. init state xywh img or something else. please don't init cuda model in this way, cuda model 
#  shold init outside of initialize, initialize means change target or init in new video, not init model.

# func(track_neighbor) 
#  watch different of track_neighbor and track, track_neighbor cannot update position,center,template,...,etc. if any code will #  change answer of call track_neighbor when input same image and xywh please put it on update_center. 

# func(update_center)
#  update teplate, DIMP, center, learningrate, train model, etc. please put it on this way. 

class NCCTracker(object):

    def __init__(self, image, region):
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]), max_val



class NCCTracker_neighbor(object):

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
        
        return xywh,max_val,neighbors_xywh, neighbors_value_only
    
    def update_center(self,xywh):
        x,y,w,h = xywh
        self.position = (x+float(w)/2,y+float(h)/2)

