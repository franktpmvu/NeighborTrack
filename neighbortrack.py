import os
import cv2
import yaml
import numpy as np
import copy
import sys
import torch
import torch.nn.functional as F
from munkres import Munkres, print_matrix

prj_path = os.path.join(os.path.dirname(__file__), '.')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from NTutils.utils import poly_iou,poly_iou_list,poly_iou_list_matrix,xy_wh_2_rect,poly_iou_list_lefttop,myerror

from NTutils.KalmanFilter import KalmanFilter
from NTutils.siammaskDetectUtils import softnms_cpu_torch




class neighbortrack(object):
    #tracker need 3 function  tracker.initialize , tracker.track_neighbor , tracker.update_center
    #and need use rev tracker ,whitch is independent same tracker to reverse track obj

    def __init__(self,tracker,im,target_pos,target_sz,revtracker=None,invtracker=None):
        #setting
        self.warning_rev_th=0.4 #0.4 IOU < W_rev then goto method
        self.rev_error_th=0.0
        self.use_neighbor_trajectory=1 # trajectory pool use neighbor trajectory, if 0 = just candidates and target history traj.
        self.rev_frames=9 #9 = reverse step tau
        self.check_probe_rev_frames=self.rev_frames # check main tracker's cycle consistency  with x frame step
        self.kalman_neighbor_mode=1 # add kalman pred to candidates
        self.time_dilate=0 # frame time dilate, if dilate = 1 then reverse = t-1,t-3,t-5,... 
        self.gt=None
        self.nms_th=0.25
        self.nms_sigma=0.01
        self.delay_start=False # if old image frames < rev frames then system will pass
        self.pred_margin_ratio=1.0
        self.neighbor_th=0.7 #alpha in paper
        self.printinfo=False
        self.cut_imageborder = False 
        #---candidate add or del----
        self.ls_add_mode = 0 # if IOU (targetHist,C_j) less than (targetHist,other candidate) , add C_j to neighbor trajectory
        self.del_winner = False # if IOU (targetHist,C_j) less than (targetHist,other candidate), del winnner traj from neighbor trajectory
        if not revtracker is None:
            self.revtracker=revtracker
        elif not invtracker is None:
            self.revtracker=invtracker
        else:
            #self.revtracker=tracker
            raise myerror('error no reverse tracker detect .')


        #
        
        self.now_frame_index=0
        self.state=self.state_init(im,target_pos,target_sz)
        self.tracker=tracker

            
        # maybe useless 
        self.ignore_frames=0 # ignore first %d frames

        
        
    def scorematrix2costmatrix(self,scorematrix):
        #change score to cost
        now_max_iou = np.max(scorematrix)
        cost_matrix = []
        for row in scorematrix:
            cost_row = []
            for col in row:
                cost_row += [now_max_iou - col]
            cost_matrix += [cost_row]


        cost_matrix = np.array(cost_matrix)

        if np.size(cost_matrix,0)-np.size(cost_matrix,1)>0:
            cost_matrix = np.pad(cost_matrix,((0,0),(0,np.size(cost_matrix,0)-np.size(cost_matrix,1))),'constant',constant_values=(now_max_iou,now_max_iou))
        if np.size(cost_matrix,0)-np.size(cost_matrix,1)<0:
            cost_matrix = np.pad(cost_matrix,((0,np.size(cost_matrix,1)-np.size(cost_matrix,0)),(0,0)),'constant',constant_values=(now_max_iou,now_max_iou))

        return cost_matrix

        
    def state_init_revobj(self, im, target_pos, target_sz):
        # in: whether input infrared image
        state = dict()
        # epoch test
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['imnow'] = im

        return state
    
    def track_rev(self,im,old_im,target_pos,target_sz,now_frame_index,rev_frame_step=None):
        # init reverse obj state from candidate and reverse it n step

        _now_frame_index = copy.copy(now_frame_index)
        
        state_revobj = self.state_init_revobj(im, target_pos, target_sz)
        if self.printinfo:
            print('---rev init target_pos sz---')
            print('target_pos = ',target_pos)
            print('target_sz = ',target_sz)
            print('------                   ---')
            
        pos = []
        sz = []
        score = []
        posind=[]
        for ind in range(len(old_im)):
            # normal len(old_im) = reverse step n=self.rev_frames, if just starting video, len(old_im) will less than reverse step n.
            if rev_frame_step is not None:
                if ind >= rev_frame_step:
                    pos.insert(0,[])
                    sz.insert(0,[])
                    score.insert(0,[])
                    posind.insert(0,[])
                    continue

            if ind == 0:
                location = xy_wh_2_rect(target_pos, target_sz)

                init_info = {'init_bbox':location}

                self.revtracker.initialize(im, init_info)

            state_revobj = self.track_state(state_revobj,old_im[-1-ind],rev_tracker = self.revtracker)
            pos.insert(0,copy.deepcopy(state_revobj['target_pos']))
            sz.insert(0,copy.deepcopy(state_revobj['target_sz']))
            score.insert(0,state_revobj['score'])
            posind.insert(0,_now_frame_index-ind-1)
                
        return pos,sz,score,posind
    
    
    def rev_neighbor_single(self,im,oldim,location_neighbor_pos,location_neighbor_sz,state \
                            ,now_frame_index):
        # each candidates to track_rev , get output reverse track answer of n frame 
        l_pos=[]
        l_sz=[]
        l_posind=[]
        l_score=[]
        for ind in range(len(location_neighbor_pos)):
            # reverse frames from model output not miximize target pos
            n_rev_pos, n_rev_sz, n_rev_score,n_rev_index = self.track_rev(im,oldim,np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),now_frame_index = now_frame_index)
            l_pos.append(n_rev_pos.copy())
            l_sz.append(n_rev_sz.copy())
            l_posind.append(n_rev_index.copy())
            l_score.append(n_rev_score.copy())
            
            if self.printinfo:

                print('--------')
                print('location_neighbor_pos[ind] = ',location_neighbor_pos[ind])
                print('ind = ',ind)
                print('n_rev_pos = ',n_rev_pos)
                print('n_rev_sz = ',n_rev_sz)
                print('--------')

        return l_pos,l_sz,l_posind,l_score


        
    def state_init(self,im, target_pos,target_sz,dataname=None,resume=None,warning_rev_th=None,gt=None):
        state=dict()
            
        state['delay_start']=self.delay_start
        state['neighbor_th'] = self.neighbor_th

        
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        state['imnow'] = im
        state['old_neighbor_pos']=[]
        state['old_neighbor_sz']=[]
        state['old_neighbor_posind']=[]
        if not gt is None:
            state['gt'] = gt
        
        state['toc_track2pass'] = 0
        state['toc_neighbor_nms'] = 0
        state['toc_revneighbor'] = 0
        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

        KF = KalmanFilter(0.1, 1, 1, 1, 1.0/20,1.0/20,w_max=state['im_w'],h_max=state['im_h'])
        #(kf_pred_x, kf_pred_y) = KF.predict()
        #(kf_estimate_x, kf_estimate_y) = KF.update(centers[0])
        # centers = Measured Position
        # kf_pred_x,y = kalman predict xy
        # kf_estimate_x,y = kalman estimate xy
        
        
        

        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['choose']='-1'
        _ = KF.update(np.array([target_pos[0],target_pos[1]]))
        state['KF'] = KF
        
        return state
    
    def track_state(self, state,im,rev_tracker=None):
        #forward tracking and get candidates

        if not rev_tracker is None:
            xywh , score , nxywh , nscore = rev_tracker.track_neighbor(im,self.neighbor_th)
            rev_tracker.update_center(xywh)
        else:
            xywh , score , nxywh , nscore = self.tracker.track_neighbor(im,self.neighbor_th)
        
        target_pos = xywh[:2]
        target_sz = xywh[2:]
        target_score = score
        if self.cut_imageborder:
            target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
            target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
            target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
            target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = target_score
        
        if 'oldim' in state:
            if len(state['oldim']) < self.rev_frames:
                state['oldim'].append(state['imnow'].copy())
            elif self.rev_frames==0:
                state['oldim']=[]
            elif len(state['oldim']) > self.rev_frames:
                state['oldim'] = state['oldim'][(len(state['oldim'])-self.rev_frames):]
            else:
                for ind in range(self.rev_frames-1):
                    state['oldim'][ind] = state['oldim'][ind+1]
                state['oldim'][self.rev_frames-1] = state['imnow'].copy()
        elif self.rev_frames==0:
            state['oldim'] = []
        else:
            state['oldim'] = []
            state['oldim'].append(state['imnow'].copy())
            
        state['imnow'] = im
        n_target_pos = [x[:2] for x in nxywh]
        n_target_sz = [x[2:] for x in nxywh]

        

        for index_n_pos_sz in range(len(n_target_pos)):
            ntpos = n_target_pos[index_n_pos_sz]
            ntsz = n_target_sz[index_n_pos_sz]
            if self.cut_imageborder:
                ntpos[0] = max(0, min(state['im_w'], ntpos[0]))
                ntpos[1] = max(0, min(state['im_h'], ntpos[1]))
                ntsz[0] = max(10, min(state['im_w'], ntsz[0]))
                ntsz[1] = max(10, min(state['im_h'], ntsz[1]))

        state['n_target_pos']=n_target_pos
        state['n_target_sz']=n_target_sz
        state['n_score']=nscore

        return state


    def track_update_history(self, state, im, gt=None,rev_frames=1,time_dilate=0,now_frame_index = 0):
        #tracking new frame and update target historical path
        _now_frame_index = copy.copy(now_frame_index)
        #norevKF
        state_old_pos = copy.copy(state['target_pos'])
        state_old_sz = copy.copy(state['target_sz'])
        state_old_score = copy.copy(state['score']) if 'score' in state else 1
        if 'use_flag' in state:
            addlengh = sum(np.array(state['use_flag'])==0)
        else:
            addlengh=0

        forward_state = self.track_state(state,im)
        
        forward_state['delay_start_flag']=1
        if 'old_target_pos' in forward_state and not rev_frames==0:
            if len(forward_state['old_target_pos']) < rev_frames+addlengh:
                if forward_state['delay_start']:
                    forward_state['delay_start_flag']=0
                forward_state['old_target_pos'].append(state_old_pos)
                forward_state['old_target_sz'].append(state_old_sz)
                forward_state['old_target_score'].append(state_old_score)
                forward_state['old_target_posind'].append(_now_frame_index-1)
            else:

                if len(forward_state['old_target_pos']) > rev_frames+addlengh:
                    forward_state['old_target_pos'] = forward_state['old_target_pos'][(len(forward_state['old_target_pos'])-rev_frames-addlengh):]
                    forward_state['old_target_sz'] = forward_state['old_target_sz'][(len(forward_state['old_target_sz'])-rev_frames-addlengh):]
                    forward_state['old_target_score'] = forward_state['old_target_score'][(len(forward_state['old_target_score'])-rev_frames-addlengh):]
                    forward_state['old_target_posind'] = forward_state['old_target_posind'][(len(forward_state['old_target_posind'])-rev_frames-addlengh):]
                    
                for ind in range(rev_frames+addlengh-1):
                    forward_state['old_target_pos'][ind] = forward_state['old_target_pos'][ind+1]
                    forward_state['old_target_sz'][ind] = forward_state['old_target_sz'][ind+1]
                    forward_state['old_target_score'][ind] = forward_state['old_target_score'][ind+1]
                    forward_state['old_target_posind'][ind] = forward_state['old_target_posind'][ind+1]

                forward_state['old_target_pos'][rev_frames+addlengh-1] = state_old_pos
                forward_state['old_target_sz'][rev_frames+addlengh-1] = state_old_sz
                forward_state['old_target_score'][rev_frames+addlengh-1] = state_old_score
                forward_state['old_target_posind'][rev_frames+addlengh-1] = _now_frame_index-1
            
        else:
            if forward_state['delay_start']:
                forward_state['delay_start_flag']=0

            forward_state['old_target_pos'] = []
            forward_state['old_target_pos'].append(state_old_pos)
            forward_state['old_target_sz'] = []
            forward_state['old_target_sz'].append(state_old_sz)
            forward_state['old_target_score'] = []
            forward_state['old_target_score'].append(state_old_score)
            forward_state['old_target_posind'] = []
            forward_state['old_target_posind'].append(_now_frame_index-1)
        
        if not rev_frames==0:
            if 'gt' in forward_state:
                if 'old_gt' in forward_state:
                    if len(forward_state['old_gt']) < rev_frames+addlengh:
                        forward_state['old_gt'].append(forward_state['gt'])

                    else:
                        if len(forward_state['old_gt']) > rev_frames+addlengh:
                            forward_state['old_gt'] = forward_state['old_gt'][(len(forward_state['old_gt'])-rev_frames-addlengh):]
                        for ind in range(rev_frames+addlengh-1):
                            forward_state['old_gt'][ind] = forward_state['old_gt'][ind+1]
                        forward_state['old_gt'][rev_frames+addlengh-1] = forward_state['gt']
                else:
                    forward_state['old_gt'] = []
                    forward_state['old_gt'].append(forward_state['gt'])

                if 'old_gt' in forward_state or 'old_target_pos' in forward_state:
                    if not 'old_gt' in forward_state:
                        if not 'gt' in forward_state:
                            print('gt is not in forward_state')
                        print(forward_state['old_target_pos'])
                        #print(state)
                        raise myerror('old_gt error1')
                    if not len(forward_state['old_gt'])==len(forward_state['old_target_pos']):
                        print(len(forward_state['old_gt']))
                        print(len(forward_state['old_target_pos']))
                        raise myerror('old_gt error2')
        forward_state['pred_target_pos'] = forward_state['target_pos']
        forward_state['pred_target_sz'] = forward_state['target_sz']
        forward_state['pred_target_score'] = forward_state['score']
        #(kf_pred_x,kf_pred_y) = forward_state['KF'].predict()
        #kf_pred_pos = np.squeeze(np.asarray([kf_pred_x,kf_pred_y]))[0]
        kf_pred_pos = forward_state['KF'].predict()
        kf_pred_pos = np.array(kf_pred_pos)
        forward_state['KF_predict_pos'] = kf_pred_pos

        return forward_state
    
        
    def _neighbor_track(self, im):
        
        #----------------------------- flag setting ---------------------------------
        state = self.state

        
        _warning_rev_th = self.warning_rev_th
        

        kalman_neighbor_mode = self.kalman_neighbor_mode
        ignore_frames = self.ignore_frames
        pred_margin_ratio = self.pred_margin_ratio
        use_neighbor_trajectory = self.use_neighbor_trajectory
        rev_error_th = self.rev_error_th
        now_frame_index = self.now_frame_index
        ls_add_mode = self.ls_add_mode
        time_dilate = self.time_dilate
        gt = self.gt
        neighbor_th = self.neighbor_th
        state['change_from_neighbor']=0
        state['change_alliou0']=0
        
        winner_index = -1
        KF_index = -1
        del_index_list=[]

        
        
        
        #  track pred 
        state = self.track_update_history(state, im,rev_frames=self.rev_frames,time_dilate=self.time_dilate,now_frame_index=now_frame_index)
        


        location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
        location_pred = xy_wh_2_rect(state['pred_target_pos'], state['pred_target_sz'])
        score_pred = copy.deepcopy(state['pred_target_score'])
        
        
        #check have candidate in image
        if 'n_target_pos' in state:
            location_neighbor_pred = []
            location_neighbor_pos = []
            location_neighbor_sz = []
            location_neighbor_posind=[]
            location_neighbor_score=[]
            location_neighbor_xyxy = []
            
            for ind in range(len(state['n_target_pos'])):
                rectind = xy_wh_2_rect([state['n_target_pos'][ind][0],state['n_target_pos'][ind][1]], [state['n_target_sz'][ind][0],state['n_target_sz'][ind][1]])
                location_neighbor_pred.append(rectind)

                location_neighbor_pos.append([state['n_target_pos'][ind][0],state['n_target_pos'][ind][1]])
                location_neighbor_sz.append([state['n_target_sz'][ind][0],state['n_target_sz'][ind][1]])
                location_neighbor_posind.append(now_frame_index)
                location_neighbor_score.append(state['n_score'])
                location_neighbor_xyxy.append([rectind[0],rectind[1],rectind[0]+rectind[2], \
                                                   rectind[1]+rectind[3],state['n_score'][ind]])

            location_neighbor_xyxy.append([location_pred[0],location_pred[1],location_pred[0]+location_pred[2], \
                                               location_pred[1]+location_pred[3],1.01])

            location_neighbor_xyxy = torch.tensor(location_neighbor_xyxy)
            
            #print('location_neighbor_xyxy = ',location_neighbor_xyxy)
            
            nms_bbox_scores,nms_bbox_index = softnms_cpu_torch(location_neighbor_xyxy,score_threshold=self.nms_th,output_index=1,sigma=self.nms_sigma)
            
            if max(nms_bbox_index) == len(location_neighbor_xyxy)-1:
                argmaxnmsbbox = np.argmax(nms_bbox_index)
                del nms_bbox_scores[argmaxnmsbbox]
                del nms_bbox_index[argmaxnmsbbox]
            else:
                print('nms_bbox_scores = ',nms_bbox_scores)
                print('nms_bbox_index = ',nms_bbox_index)
                raise myerror('nms_bbox error ')
                


            location_neighbor_pred = [location_neighbor_pred[ind] for ind in nms_bbox_index]
            location_neighbor_pos = [location_neighbor_pos[ind] for ind in nms_bbox_index]                
            location_neighbor_posind = [location_neighbor_posind[ind] for ind in nms_bbox_index]                
            location_neighbor_score = [location_neighbor_score[ind] for ind in nms_bbox_index]                
            location_neighbor_sz = [location_neighbor_sz[ind] for ind in nms_bbox_index]


        
        #if candidate(without C_j) + trajectory pool >0
        
        if len(location_neighbor_pos)+len(state['old_neighbor_pos'])>0 and self.rev_frames!=0:
            
            state_old_im = state['oldim'].copy()
            # reverse frames from model output maximize target pos
                
            rev_target_pos, rev_target_sz, rev_score,rev_index = self.track_rev(im,state_old_im,state['target_pos'], state['target_sz'],now_frame_index = now_frame_index,rev_frame_step=self.check_probe_rev_frames)
            

                
                
            
            b_overlap = poly_iou_list(state['old_target_pos'][::1+self.time_dilate][ignore_frames:],state['old_target_sz'][::1+self.time_dilate][ignore_frames:],rev_target_pos[::1+self.time_dilate][ignore_frames:],rev_target_sz[::1+self.time_dilate][ignore_frames:])

            
            
            
            state['rev_target_pos'] = rev_target_pos
            state['rev_target_sz'] = rev_target_sz
            state['rev_target_score'] = rev_score
            state['rev_target_posind'] = rev_index
            
            state['reviou'] = b_overlap
            state['reviou_ori'] = b_overlap
            
            if self.printinfo:

                print('---old target pos sz  -------')
                print(state['old_target_pos'][::1+self.time_dilate][ignore_frames:])
                print(state['old_target_sz'][::1+self.time_dilate][ignore_frames:])

                print('---rev target pos sz  -------')
                print(rev_target_pos[::1+self.time_dilate][ignore_frames:])
                print(rev_target_sz[::1+self.time_dilate][ignore_frames:])
                print('---b overlap  -------')

                print('b_overlap = ',b_overlap)
                print('------  -------')

            
        else:
            state['rev_target_pos'] = []
            state['rev_target_sz'] = []
            state['rev_target_score'] = []
            state['rev_target_posind'] = []
            
            state['reviou'] = 1.01
            

        location_KF_pred = xy_wh_2_rect(state['KF_predict_pos'], state['old_target_sz'][-1])
        
        
        all_rev_iou=[]
        state['choose']='-1'
        if state['reviou'] < _warning_rev_th and state['delay_start_flag']==1:
            
            if kalman_neighbor_mode:
                location_neighbor_pos.append(state['KF_predict_pos'])
                location_neighbor_sz.append(state['old_target_sz'][-1])
                location_neighbor_posind.append(now_frame_index)
                location_neighbor_score.append(0)
                KF_index = len(location_neighbor_pos)-1

            costmatrix=np.zeros((len(state['old_neighbor_pos'])+1,len(location_neighbor_pos)+1))
            
            n_rev_pred_rect=[]
            n_rev_pred_pos=[]
            n_rev_pred_sz=[]
            n_rev_pred_posind=[]
            n_rev_pred_score=[]






            l_pos_,l_sz_,l_posind_,l_score_=\
            self.rev_neighbor_single(im,state['oldim'],location_neighbor_pos,location_neighbor_sz,\
                                     state,now_frame_index)
            if self.printinfo:

                print('-------')
                print('location_neighbor_pos = ',location_neighbor_pos)
                print('l_pos_ = ',l_pos_)
                print('l_sz_ = ',l_sz_)
                print('l_posind_ = ',l_posind_)
                print('l_score_ = ',l_score_)
                print('-------')





            for ind in range(len(location_neighbor_pos)):
                #reverse frames from model output not miximize target pos

                n_rev_pred_pos.append(l_pos_[ind].copy())
                n_rev_pred_sz.append(l_sz_[ind].copy())
                n_rev_pred_posind.append(l_posind_[ind].copy())
                n_rev_pred_score.append(l_score_[ind].copy())

                l_pos_[ind].append(np.array(location_neighbor_pos[ind]))
                l_sz_[ind].append(np.array(location_neighbor_sz[ind]))
                l_posind_[ind].append(np.array(location_neighbor_posind[ind]))
                l_score_[ind].append(np.array(location_neighbor_score[ind]))




                location_neighbor_pos[ind] = l_pos_[ind] # now+rev
                location_neighbor_sz[ind] = l_sz_[ind]
                location_neighbor_posind[ind] = l_posind_[ind]
                location_neighbor_score[ind] = l_score_[ind]


            all_rev_iou.append(state['reviou'])

            if self.printinfo:

                print('--')
                print('n_rev_pred_pos = ',n_rev_pred_pos)
                print('--')

            b_overlap_npred_rev = [poly_iou_list(state['old_target_pos'][::1+self.time_dilate][ignore_frames:],state['old_target_sz'][::1+self.time_dilate][ignore_frames:],n_rev_pred_pos[ind][::1+self.time_dilate][ignore_frames:],n_rev_pred_sz[ind][::1+self.time_dilate][ignore_frames:]) for ind in range(len(n_rev_pred_pos))]



            for b_o_n in b_overlap_npred_rev:
                all_rev_iou.append(b_o_n)
            if self.printinfo:

                print('------------------------------------')
                print('all_rev_iou = ',all_rev_iou)
                print('all_rev_iou[0] = ',all_rev_iou[0])
                print('pred_margin_ratio = ',pred_margin_ratio)
                print('all_rev_iou[0] = ',all_rev_iou[0]*pred_margin_ratio)
                print('location_neighbor_pos = ',location_neighbor_pos)
                print('location_neighbor_sz = ',location_neighbor_sz)

                print('------------------------------------')

            all_rev_iou[0]=all_rev_iou[0]*pred_margin_ratio

            if use_neighbor_trajectory:
                costmatrix[0,]=all_rev_iou
                _n_rev_pred_pos = []
                _n_rev_pred_sz = []
                _n_rev_pred_posind = []

                _n_rev_pred_pos.append(state['rev_target_pos'])
                _n_rev_pred_sz.append(state['rev_target_sz'])
                _n_rev_pred_posind.append(state['rev_target_posind'])

                for ind in range(len(n_rev_pred_pos)):
                    _n_rev_pred_pos.append(n_rev_pred_pos[ind])
                    _n_rev_pred_sz.append(n_rev_pred_sz[ind])
                    _n_rev_pred_posind.append(n_rev_pred_posind[ind])

                n_rev_pred_pos = _n_rev_pred_pos
                n_rev_pred_sz = _n_rev_pred_sz
                n_rev_pred_posind = _n_rev_pred_posind


                state['old_neighbor_pos'] = [x[1:] for x in state['old_neighbor_pos'] if len(x)>self.rev_frames]
                state['old_neighbor_sz'] = [x[1:] for x in state['old_neighbor_sz'] if len(x)>self.rev_frames]
                state['old_neighbor_posind'] = [x[1:] for x in state['old_neighbor_posind'] if len(x)>self.rev_frames]


                for ind in range(len(state['old_neighbor_pos'])):

                    try:

                        b_overlap_nold_rev = [poly_iou_list(state['old_neighbor_pos'][ind][::1+self.time_dilate][ignore_frames:],state['old_neighbor_sz'][ind][::1+self.time_dilate][ignore_frames:],n_rev_pred_pos[ind2][::1+self.time_dilate][ignore_frames:],n_rev_pred_sz[ind2][::1+self.time_dilate][ignore_frames:]) for ind2 in range(len(n_rev_pred_pos))]
                    except:
                        print('errv1 = ',state['old_neighbor_pos'][ind][::1+self.time_dilate][ignore_frames:])
                        print('state[old_neighbor_posind][ind][::1+self.time_dilate][ignore_frames:] = ',state['old_neighbor_posind'][ind][::1+self.time_dilate][ignore_frames:])

                        for ind2 in range(len(n_rev_pred_pos)):
                            print('n_rev_pred_posind[ind2][::1+self.time_dilate][ignore_frames:] = ',n_rev_pred_posind[ind2][::1+self.time_dilate][ignore_frames:])
                        raise myerror('b_overlap error1')

                    costmatrix[ind+1,]=b_overlap_nold_rev

                cost_matrix = self.scorematrix2costmatrix(costmatrix)
                #from munkres import Munkres, print_matrix

                Mun = Munkres()
                Munindexes = Mun.compute(cost_matrix)

                minusmatrix = copy.deepcopy(costmatrix)*-1
                row, column = Munindexes[0]


                try:
                    value = costmatrix[row][column]
                    minusmatrix[row][column] = value
                except:
                    pass
                all_rev_iou = minusmatrix[0,:]









            all_rev_iou = np.array(all_rev_iou)

            if all_rev_iou.max()>rev_error_th:
                if self.printinfo:
                    print('all_rev_iou.argmax() = ',all_rev_iou.argmax())
                    print('all_rev_iou = ',all_rev_iou)

                if all_rev_iou.argmax()==0:# max IOU = C_j
                    (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                    state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                    state['choose']='model pred1'


                else:
                    state['change_from_neighbor']=1
                    winner_index=all_rev_iou.argmax()-1# 0 is model pred so -1 = index of location_neighbor_pos
                    state['winner_rev_pos']=location_neighbor_pos[winner_index][:-1].copy()
                    state['winner_rev_sz']=location_neighbor_sz[winner_index][:-1].copy()
                    #need delete winner_rev_pos
                    if self.del_winner:
                        if all_rev_iou.argmax()==(KF_index+1):
                            del_index_list.append(winner_index)

                    if ls_add_mode:
                        rev_pos = state['rev_target_pos'].copy()
                        rev_sz = state['rev_target_sz'].copy()
                        rev_posind = state['rev_target_posind'].copy()

                        rev_pos.append(state['target_pos'])
                        rev_sz.append(state['target_sz'])
                        rev_posind.append(now_frame_index)

                        if len(location_neighbor_pos)==0:
                            location_neighbor_pos = np.array([rev_pos])
                            location_neighbor_sz = np.array([rev_sz])
                            location_neighbor_posind = np.array([rev_posind])
                        else:
                            location_neighbor_pos = np.append(location_neighbor_pos,[rev_pos],0)
                            location_neighbor_sz = np.append(location_neighbor_sz,[rev_sz],0)
                            location_neighbor_posind = np.append(location_neighbor_posind,[rev_posind],0)

                    if kalman_neighbor_mode and all_rev_iou.argmax()==(KF_index+1) :#0 is model pred so +1 == KF_index 

                        state['target_pos'] = state['KF_predict_pos']
                        state['target_sz'] = state['old_target_sz'][-1]
                        state['KF_est_pos'] = state['target_pos']
                        state['choose']='KF1'

                    else:

                        now_max_iou_ind = nms_bbox_index[all_rev_iou.argmax()-1]

                        state['target_pos'][0] = state['n_target_pos'][now_max_iou_ind][0]
                        state['target_pos'][1] = state['n_target_pos'][now_max_iou_ind][1]

                        state['target_sz'][0] = state['n_target_sz'][now_max_iou_ind][0]
                        state['target_sz'][1] = state['n_target_sz'][now_max_iou_ind][1]

                        state['score'] = copy.deepcopy(state['n_score'][now_max_iou_ind])
                        (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                        state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                        state['choose']='neighbor0'
            else: #if all of candidate IOU ==0 then choose KF answer
                if ls_add_mode:
                    rev_pos = state['rev_target_pos'].copy()
                    rev_sz = state['rev_target_sz'].copy()
                    rev_posind = state['rev_target_posind'].copy()

                    rev_pos.append(state['target_pos'])
                    rev_sz.append(state['target_sz'])
                    rev_posind.append(now_frame_index)






                    if len(location_neighbor_pos)==0:
                        location_neighbor_pos = np.array([rev_pos])
                        location_neighbor_sz = np.array([rev_sz])
                        location_neighbor_posind = np.array([rev_posind])
                    else:

                        location_neighbor_pos = np.append(location_neighbor_pos,[rev_pos],0)
                        location_neighbor_sz = np.append(location_neighbor_sz,[rev_sz],0)
                        location_neighbor_posind = np.append(location_neighbor_posind,[rev_posind],0)




                state['change_alliou0']=1
                state['target_pos'] = state['KF_predict_pos']
                state['target_sz'] = state['old_target_sz'][-1]
                state['KF_est_pos'] = state['target_pos']
                state['choose']='KF2'
                    
        else:
            state['choose']='model pred2'
            (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
            state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
            
            if len(location_neighbor_pos)>0:
                l_pos_,l_sz_,l_posind_,l_score_=\
                 self.rev_neighbor_single(im,state['oldim'],location_neighbor_pos,location_neighbor_sz,\
                  state,now_frame_index)

                for ind in range(len(location_neighbor_pos)):                        
                    l_pos_[ind].append(np.array(location_neighbor_pos[ind]))
                    l_sz_[ind].append(np.array(location_neighbor_sz[ind]))
                    l_posind_[ind].append(np.array(location_neighbor_posind[ind]))
                    
                    location_neighbor_pos[ind] = l_pos_[ind]
                    location_neighbor_sz[ind] = l_sz_[ind]
                    location_neighbor_posind[ind] = l_posind_[ind]
        
        
        
        if self.printinfo:
            if not 'model pred2' in state['choose']:
                print('state[choose] = ',state['choose'])

        if kalman_neighbor_mode and not KF_index==-1:
            if not KF_index in del_index_list:
                del_index_list.append(KF_index)
                if self.printinfo:
                    print('now add KF index in del_index_list')
                    print(del_index_list)
        if del_index_list:
            #print('state[choose] = ',state['choose'])
            
            #if state['change_from_neighbor']==1 and state['choose']=='neighbor0':
            #    print(state['target_pos'])
                
            if type(location_neighbor_pos) is np.ndarray:
                location_neighbor_pos = location_neighbor_pos.tolist()
            if type(location_neighbor_sz) is np.ndarray:
                location_neighbor_sz = location_neighbor_sz.tolist()
            if type(location_neighbor_posind) is np.ndarray:
                location_neighbor_posind = location_neighbor_posind.tolist()
            if type(location_neighbor_score) is np.ndarray:
                location_neighbor_score = location_neighbor_score.tolist()
                
            del_list_revsort = sorted(del_index_list, reverse = True)
            for revdelindex in del_list_revsort:
                del location_neighbor_pos[revdelindex]
                del location_neighbor_sz[revdelindex]
                del location_neighbor_posind[revdelindex]
                del location_neighbor_score[revdelindex]
            #if state['change_from_neighbor']==1 and state['choose']=='neighbor0':
            #    print(state['target_pos'])
            #    print(state['choose'])
            #    dsa

        if 'old_neighbor_pos' in state.keys():
            state['trackpool_neighbor_pos']=copy.deepcopy(state['old_neighbor_pos'])
        if 'old_neighbor_sz' in state.keys():
            state['trackpool_neighbor_sz']=copy.deepcopy(state['old_neighbor_sz'])
        if 'old_neighbor_posind' in state.keys():
            state['trackpool_neighbor_posind']=copy.deepcopy(state['old_neighbor_posind'])
        if 'old_neighbor_score' in state.keys():
            state['trackpool_neighbor_score']=copy.deepcopy(state['old_neighbor_score'])

            
        state['old_neighbor_pos']=copy.deepcopy(location_neighbor_pos)#candidate
        state['old_neighbor_sz']=copy.deepcopy(location_neighbor_sz)
        state['old_neighbor_posind']=copy.deepcopy(location_neighbor_posind)
        state['old_neighbor_score']=copy.deepcopy(location_neighbor_score)


        state['location_neighbor_pred'] = location_neighbor_pred
        state['location_pred'] = location_pred
        state['score_pred'] = score_pred
        state['location_KF_pred'] = location_KF_pred
        state['all_rev_iou'] = all_rev_iou
        if not gt is None:
            state['gt']=gt
        if self.cut_imageborder:
            location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
        else:
            location = [float(state['target_pos'][0]),float(state['target_pos'][1]), float(state['target_sz'][0]),float(state['target_sz'][1])]

        x=location[0]
        y=location[1]
        w=location[2]
        h=location[3]
        #print('center update last')
        self.tracker.update_center([x,y,w,h])
        
        return state