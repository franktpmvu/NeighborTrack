import os
import cv2
import yaml
import numpy as np
import copy

import torch
import torch.nn.functional as F
from utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round,cxy_wh_2_rect,poly_iou,poly_iou_list,poly_iou_list_matrix
from utils.KalmanFilter import KalmanFilter
from utils.siammaskDetectUtils import softnms_cpu_torch
from munkres import Munkres, print_matrix



class Ocean2pass(object):
    def __init__(self, info):
        super(Ocean2pass, self).__init__()
        self.info = info   # model and benchmark info
        self.stride = 8
        self.align = info.align
        self.online = info.online
        self.trt = info.TRT

    def init_invobj(self, im, target_pos, target_sz, model, p):
        # in: whether input infrared image
        state = dict()
        # epoch test
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        state['im'] = im
        
        #self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = z_crop.unsqueeze(0)
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones((int(p.score_size), int(p.score_size)))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['imnow'] = im
        state['z'] = z.cuda()

        return state

        
        
    def init(self, im, target_pos, target_sz, model, hp=None,online_tracker_inv=None,dataname=None,resume=None,warning_inv_th=None,gt=None):
        # in: whether input infrared image
        state = dict()
        if warning_inv_th=='dy':
            state['historyCCinv']=[1]
            state['CCmean']=1
            state['CCstd']=0

        # epoch test
        p = OceanConfig()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        state['imnow'] = im
        state['old_neighbor_pos']=[]
        state['old_neighbor_sz']=[]
        state['old_neighbor_posind']=[]

        state['warning_2pass'] = 0
        state['choose_right_pred'] = 0
        state['max_iou_in_model_pred'] = 0
        state['max_iou_in_kalman_pred'] = 0
        state['max_iou_in_neighbor_pred'] = 0
        state['flag_output_image']=0
        state['nowconsecutive']=0
        if not gt is None:
            state['gt'] = gt
        if online_tracker_inv is not None:
            print('init online_tracker_inv')
            state['online_tracker_inv'] = online_tracker_inv
            state['dataname'] = dataname
            state['resume'] = resume
        #print('online_tracker_inv = ',online_tracker_inv)
        


        state['toc_track2pass'] = 0
        state['toc_neighbor_nms'] = 0
        state['toc_invneighbor'] = 0

        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        KF = KalmanFilter(0.1, 1, 1, 1, 1.0/20,1.0/20)
        #(kf_pred_x, kf_pred_y) = KF.predict()
        #(kf_estimate_x, kf_estimate_y) = KF.update(centers[0])
        # centers = Measured Position
        # kf_pred_x,y = kalman predict xy
        # kf_estimate_x,y = kalman estimate xy
        
        # single test
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT', 'GOT10K', 'LASOT'] if x in self.info.dataset]
            if len(prefix) == 0: prefix = [self.info.dataset]
            absPath = os.path.abspath(os.path.dirname(__file__))
            yname = 'Ocean.yaml'
            yamlPath = os.path.join(absPath, '../../experiments/test/{0}/'.format(prefix[0]), yname)
            cfg = load_yaml(yamlPath)
            if self.online:
                temp = self.info.dataset + 'ON'
                cfg_benchmark = cfg[temp]
            else:
                cfg_benchmark = cfg[self.info.dataset]
            p.update(cfg_benchmark)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()

        # double check
        #print('======= hyper-parameters: penalty_k: {}, wi: {}, lr: {}, ratio: {}, instance_sz: {}, score_sz: {} ======='.format(p.penalty_k, p.window_influence, p.lr, p.ratio, p.instance_size, p.score_size))

        # param tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = hp['big_sz']
                p.renew()
            else:
                p.instance_size = hp['small_sz']
                p.renew()

        if self.trt:
            print('====> TRT version testing: only support 255 input, the hyper-param is random <====')
            p.instance_size = 255
            p.renew()

        self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = z_crop.unsqueeze(0)
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones((int(p.score_size), int(p.score_size)))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['imnow'] = im
        state['choose']='-1'
        state['z'] = z.cuda()
        _ = KF.update(np.array([target_pos[0],target_pos[1]]))
        state['KF'] = KF

        return state
    
    def update_det(self, net, x_crops, target_pos, target_sz, window, scale_z, p,threshold=0.7,score_only=False,state=None,static_th=False):
        
        if self.align:
            cls_score, bbox_pred, cls_align = net.track(x_crops)
            
            
            cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
            cls_align = F.sigmoid(cls_align).squeeze().cpu().data.numpy()
            cls_score = p.ratio * cls_score + (1- p.ratio) * cls_align
            net_track_ans = [cls_score, bbox_pred, cls_align]

            
        else:
            cls_score, bbox_pred = net.track(x_crops)
            
            cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
            net_track_ans = [cls_score, bbox_pred]


        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2-pred_x1, pred_y2-pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2-pred_x1) / (pred_y2-pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score
        #if not state is None:
        #    state['score_apperiance']=
        if not score_only:
            # window penalty
            pscore = pscore * (1 - p.window_influence) + window * p.window_influence


        if self.online_score is not None:
            s_size = pscore.shape[0]
            o_score = cv2.resize(self.online_score, (s_size, s_size), interpolation=cv2.INTER_CUBIC)
            pscore = p.online_ratio * o_score + (1 - p.online_ratio) * pscore
        else:
            pass
        #if not state is None:
        #    state['score_apperiance_online']=

        # get max
        #r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
        #print('np.where(pscore>threshold) = ',np.where(pscore>threshold))
        #print('pschore.max = ',np.max(pscore))
        #print('np.where(pscore>threshold*pscoremax) = ',np.where(pscore>threshold*np.max(pscore)))
        
        #if not np.where(pscore>threshold):#old before 211021
        if static_th:
            if not np.where(pscore>threshold):
                print('not have neighbour')
                return [], [], []
            #r_max, c_max = np.unravel_index(pscore>threshold, pscore.shape)
            r_max, c_max = np.where(pscore>threshold)
        else:
            if not np.where(pscore>threshold*np.max(pscore)):
                print('not have neighbour')
                return [], [], []
            #r_max, c_max = np.unravel_index(pscore>threshold, pscore.shape)
            r_max, c_max = np.where(pscore>threshold*np.max(pscore))

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]
        
        
        #print(res_w)
        #print(res_h)
        #print(target_sz)
        target_pos = np.array([res_xs, res_ys])
        _target_sz = lr * np.array([res_w, res_h])
        _target_sz[0] = _target_sz[0]+target_sz[0] * (1 - lr)
        _target_sz[1] = _target_sz[1]+target_sz[1] * (1 - lr)
        target_sz = _target_sz
        
        
        
        return net_track_ans,target_pos, target_sz, cls_score[r_max, c_max]

    
    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p,neighbor=False,state=None):
        if neighbor:
            det_window = np.ones((int(p.score_size), int(p.score_size)))

            net_track_ans,t_pos,t_sz,t_cls_sc = self.update_det(net, x_crops, target_pos, target_sz, det_window, scale_z, p,score_only=state['neighbor_score_only'],static_th=state['neighbor_static_th'],threshold=state['neighbor_th'])
            if self.align:
                cls_score, bbox_pred, cls_align = net_track_ans
            else:
                cls_score, bbox_pred = net_track_ans
        else:
            if self.align:
                cls_score, bbox_pred, cls_align = net.track(x_crops)

                cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
                cls_align = F.sigmoid(cls_align).squeeze().cpu().data.numpy()
                cls_score = p.ratio * cls_score + (1- p.ratio) * cls_align

            else:
                cls_score, bbox_pred = net.track(x_crops)
                cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2-pred_x1, pred_y2-pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2-pred_x1) / (pred_y2-pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        if self.online_score is not None:
            s_size = pscore.shape[0]
            o_score = cv2.resize(self.online_score, (s_size, s_size), interpolation=cv2.INTER_CUBIC)
            pscore = p.online_ratio * o_score + (1 - p.online_ratio) * pscore
        else:
            pass

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])
        if neighbor:
            #print(target_pos)
            #print(target_sz)
            #print(cls_score[r_max, c_max])
            #print(t_pos)
            #print(t_sz)
            #print(t_cls_sc)
            
            
            return target_pos, target_sz, cls_score[r_max, c_max],t_pos,t_sz,t_cls_sc
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, state, im, online_score=None, gt=None,neighbor=False,inv_frames=1):
        p = state['p']
        net = state['net']
        net.template(state['z'])
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        if online_score is not None:
            self.online_score = online_score.squeeze().cpu().data.numpy()
        else:
            self.online_score = None

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        
        #from torchvision.utils import save_image
        #import torchvision


        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        #save_image(x_crop/255, 'x_crop.png')
        #print(x_crop)
        x_crop = x_crop.unsqueeze(0)
        if neighbor:
            p.instance_size_ori = copy.copy(p.instance_size)
            p.instance_size = int(p.instance_size*state['neighbor_search_ratio'])
            p.renew()
            self.grids(p)
            
            _hc_z = target_sz[1] + p.context_amount * sum(target_sz)
            _wc_z = target_sz[0] + p.context_amount * sum(target_sz)
            _s_z = np.sqrt(_wc_z * _hc_z)
            _scale_z = p.exemplar_size / _s_z
            _d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
            _pad = _d_search / _scale_z
            _s_x = _s_z + 2 * _pad

            
            det_window = np.ones((int(p.score_size), int(p.score_size)))
            
            x_crop_neighbor, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(_s_x), avg_chans)
            #save_image(x_crop_neighbor/255, 'x_crop_neighbor.png')
            
            
            
            x_crop_neighbor = x_crop_neighbor.unsqueeze(0)

            _, n_target_pos, n_target_sz, n_target_score = self.update_det(net, x_crop_neighbor.cuda(), target_pos, target_sz*scale_z, det_window, scale_z, p,score_only=state['neighbor_score_only'],static_th=state['neighbor_static_th'],threshold=state['neighbor_th'])
            
            p.instance_size = p.instance_size_ori
            p.renew()
            self.grids(p)
            
            target_pos, target_sz, target_score = self.update(net, x_crop.cuda(), target_pos, target_sz*scale_z, window, scale_z, p,state=state)
            #print('target_pos.size = ',target_pos.size)
            #print('n_target_pos.size = ',n_target_pos.size)
            #dsadsa
            
        else:
            target_pos, target_sz, target_score = self.update(net, x_crop.cuda(), target_pos, target_sz*scale_z, window, scale_z, p,state=state)
        

        
        
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p
        state['score'] = target_score
        
        if 'oldim' in state:
            if len(state['oldim']) < inv_frames:
                state['oldim'].append(state['imnow'].copy())
            elif inv_frames==0:
                state['oldim']=[]
            elif len(state['oldim']) > inv_frames:
                state['oldim'] = state['oldim'][(len(state['oldim'])-inv_frames):]
            else:
                for ind in range(inv_frames-1):
                    state['oldim'][ind] = state['oldim'][ind+1]
                state['oldim'][inv_frames-1] = state['imnow'].copy()
        elif inv_frames==0:
            state['oldim'] = []
        else:
            state['oldim'] = []
            state['oldim'].append(state['imnow'].copy())
            
        state['imnow'] = im
        if neighbor:
            #have bug
            for index_n_pos_sz in range(len(n_target_pos)):
                ntpos = n_target_pos[index_n_pos_sz]
                ntsz = n_target_sz[index_n_pos_sz]

                ntpos[0] = max(0, min(state['im_w'], ntpos[0]))
                ntpos[1] = max(0, min(state['im_h'], ntpos[1]))
                ntsz[0] = max(10, min(state['im_w'], ntsz[0]))
                ntsz[1] = max(10, min(state['im_h'], ntsz[1]))

            state['n_target_pos']=n_target_pos
            state['n_target_sz']=n_target_sz
            state['n_score']=n_target_score

        return state
    
    
    def track_inv(self,im,old_im,target_pos,target_sz,net,p,online_score=None,gt=None,online_tracker_inv=None,dataname=None,resume=None,now_frame_index=0,inv_start_from_history=False,old_target_pos_sz=None):
        if inv_start_from_history:
            if old_target_pos_sz is None:
                print('need old_target_pos_sz to initial the start point from history!')
                dsadsa
        _now_frame_index = copy.copy(now_frame_index)



        
        state_invobj = self.init_invobj(im, target_pos, target_sz, net, p)
        
        if not online_tracker_inv is None:
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            online_tracker_inv.init(im, rgb_im, net, target_pos, target_sz, True, dataname=dataname, resume=resume)

        

        
        
        
        pos = []
        sz = []
        score = []
        posind=[]
        if inv_start_from_history:
            _old_target_pos ,_old_target_sz = old_target_pos_sz
            old_target_pos = copy.deepcopy(_old_target_pos)
            old_target_sz = copy.deepcopy(_old_target_sz)
            #print(old_target_pos)
        #print('len(old_im) = ',len(old_im))
        for ind in range(len(old_im)):
            rgb_im = cv2.cvtColor(old_im[-1-ind], cv2.COLOR_BGR2RGB)

            if not online_tracker_inv is None:
                state_invobj = online_tracker_inv.track(old_im[-1-ind], rgb_im, self, state_invobj)
                #print('now is onlineinv')

            else:
                #state_invobj = self.track(state_invobj,old_im[-1-ind],online_score=online_score,gt=gt)
                #print('now is offlineinv')
                state_invobj = self.track(state_invobj,old_im[-1-ind],online_score=None,gt=gt)
            pos.insert(0,copy.deepcopy(state_invobj['target_pos']))
            sz.insert(0,copy.deepcopy(state_invobj['target_sz']))
            score.insert(0,state_invobj['score'])
            posind.insert(0,_now_frame_index-ind-1)
            if inv_start_from_history:
                #print('ISFH start')
                #print('bf pos = ',state_invobj['target_pos'])
                state_invobj['target_pos'] = old_target_pos[-1-ind]
                state_invobj['target_sz'] = old_target_sz[-1-ind]
                #print('aft pos = ',state_invobj['target_pos'])
            
        #print('pos = ',pos)
        #print('sz = ',sz)
        #print('score = ',score)

        return pos,sz,score,posind
    
    
    def track_inv_neighbor(self,im,old_im,target_pos,target_sz,net,p,online_score=None,gt=None,online_tracker_inv=None,dataname=None,resume=None,now_frame_index=0,state=None,neighbor_score_only=False,neighbor_static_th=False,neighbor_th=0.7):

        
        if not state:
            _state_invobj = self.init_invobj(im, target_pos, target_sz, net, p)
            
            #print('not support online mode now ,too slow')
            #if not online_tracker_inv is None:
            #    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #    online_tracker_inv.init(im, rgb_im, net, target_pos, target_sz, True, dataname=dataname, resume=resume)
            #    state_invobj['online_tracker']=online_tracker_inv
            
        else:
            _state_invobj=state
            _state_invobj['target_pos'] = target_pos
            _state_invobj['target_sz'] = target_sz
            _state_invobj['net'].template(_state_invobj['z'])

        #pos = []
        #sz = []
        #score = []
        #posind=[]
        
        #rgb_im = cv2.cvtColor(old_im[-1-ind], cv2.COLOR_BGR2RGB)

        #if not online_tracker_inv is None:
        #    state_invobj = online_tracker_inv.track(old_im[-1-ind], rgb_im, self, state_invobj)
        #    #print('now is onlineinv')

        #else:
            #state_invobj = self.track(state_invobj,old_im[-1-ind],online_score=online_score,gt=gt)
            #print('now is offlineinv')
            #state_invobj = self.track(state_invobj,old_im[-1-ind],online_score=None,gt=gt)

        _state_invobj['neighbor_search_ratio']=1.0
        _state_invobj['neighbor_score_only'] = neighbor_score_only
        _state_invobj['neighbor_static_th'] = neighbor_static_th
        _state_invobj['neighbor_th'] = neighbor_th

        state_invobj = self.track(_state_invobj,old_im,online_score=None,gt=gt,neighbor=True)
        #forward_state = self.track(state,im,online_score,gt,neighbor=True,inv_frames=inv_frames+addlengh)

        #pos=copy.deepcopy(state_invobj['target_pos'])
        #sz=copy.deepcopy(state_invobj['target_sz'])
        #score=copy.deepcopy(state_invobj['score'])

        #posind=_now_frame_index-ind-1


        return state_invobj
    
    
    
    
    def track_2pass(self, state, im, online_score=None, gt=None,pass_check_mode=False,inv_frames=1,time_dilate=0,now_frame_index = 0):
        _now_frame_index = copy.copy(now_frame_index)
        #toc = 0
        #toc_track1 = 0
        #toc_inv = 0
        #toc_writemem = 0
        
        #tic_total = cv2.getTickCount()

        
        #noinvKF
        '''
        if state['choose'].find('KF')==-1:
            state_old_pos = state['target_pos'].copy()
            state_old_sz = state['target_sz'].copy()
            state_old_score = state['score'].copy() if 'score' in state else 1
        else:
            state_old_pos = state['target_pos'].copy()*0-100
            state_old_sz = state['target_sz'].copy()*0+1
            state_old_score = state['score'].copy() if 'score' in state else 0
        '''
        state_old_pos = state['target_pos'].copy()
        state_old_sz = state['target_sz'].copy()
        state_old_score = state['score'].copy() if 'score' in state else 1

        #state_old_im = state['imnow'].copy()
        
        if 'use_flag' in state:
            addlengh = sum(np.array(state['use_flag'])==0)
        else:
            addlengh=0

        #print(state)
        forward_state = self.track(state,im,online_score,gt,neighbor=True,inv_frames=inv_frames+addlengh)
        
        #toc_track1 += cv2.getTickCount() - tic_total

        
        
        #state_old_im = forward_state['oldim'].copy()

        #state_invobj = self.init_invobj(im, forward_state['target_pos'], forward_state['target_sz'], forward_state['net'], forward_state['p'])
        #state_invobj = self.track(state_invobj,state_old_im,online_score,gt)

        
        #tic = cv2.getTickCount()

        
        #inv_target_pos, inv_target_sz, inv_score = self.track_inv(im,state_old_im,forward_state['target_pos'], forward_state['target_sz'], forward_state['net'], forward_state['p'],online_score,gt)
        
        #toc_inv += cv2.getTickCount() - tic

        #tic = cv2.getTickCount()

        
        
        forward_state['delay_start_flag']=1
        if 'old_target_pos' in forward_state and not inv_frames==0:
            if len(forward_state['old_target_pos']) < inv_frames+addlengh:
                if forward_state['delay_start']:
                    forward_state['delay_start_flag']=0
                forward_state['old_target_pos'].append(state_old_pos)
                forward_state['old_target_sz'].append(state_old_sz)
                forward_state['old_target_score'].append(state_old_score)
                forward_state['old_target_posind'].append(_now_frame_index-1)
            else:

                if len(forward_state['old_target_pos']) > inv_frames+addlengh:
                    forward_state['old_target_pos'] = forward_state['old_target_pos'][(len(forward_state['old_target_pos'])-inv_frames-addlengh):]
                    forward_state['old_target_sz'] = forward_state['old_target_sz'][(len(forward_state['old_target_sz'])-inv_frames-addlengh):]
                    forward_state['old_target_score'] = forward_state['old_target_score'][(len(forward_state['old_target_score'])-inv_frames-addlengh):]
                    forward_state['old_target_posind'] = forward_state['old_target_posind'][(len(forward_state['old_target_posind'])-inv_frames-addlengh):]
                    
                for ind in range(inv_frames+addlengh-1):
                    forward_state['old_target_pos'][ind] = forward_state['old_target_pos'][ind+1]
                    forward_state['old_target_sz'][ind] = forward_state['old_target_sz'][ind+1]
                    forward_state['old_target_score'][ind] = forward_state['old_target_score'][ind+1]
                    forward_state['old_target_posind'][ind] = forward_state['old_target_posind'][ind+1]

                forward_state['old_target_pos'][inv_frames+addlengh-1] = state_old_pos
                forward_state['old_target_sz'][inv_frames+addlengh-1] = state_old_sz
                forward_state['old_target_score'][inv_frames+addlengh-1] = state_old_score
                forward_state['old_target_posind'][inv_frames+addlengh-1] = _now_frame_index-1
            
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
        
        if not inv_frames==0:
            if 'gt' in forward_state:
                if 'old_gt' in forward_state:
                    if len(forward_state['old_gt']) < inv_frames+addlengh:
                        forward_state['old_gt'].append(forward_state['gt'])

                    else:
                        if len(forward_state['old_gt']) > inv_frames+addlengh:
                            forward_state['old_gt'] = forward_state['old_gt'][(len(forward_state['old_gt'])-inv_frames-addlengh):]
                        for ind in range(inv_frames+addlengh-1):
                            forward_state['old_gt'][ind] = forward_state['old_gt'][ind+1]
                        forward_state['old_gt'][inv_frames+addlengh-1] = forward_state['gt']
                else:
                    forward_state['old_gt'] = []
                    forward_state['old_gt'].append(forward_state['gt'])

                if 'old_gt' in forward_state or 'old_target_pos' in forward_state:
                    if not 'old_gt' in forward_state:
                        if not 'gt' in forward_state:
                            print('gt is not in forward_state')
                        print(forward_state['old_target_pos'])
                        #print(state)
                        dsadsa
                    if not len(forward_state['old_gt'])==len(forward_state['old_target_pos']):
                        print(len(forward_state['old_gt']))
                        print(len(forward_state['old_target_pos']))
                        dsadsa
        #b_overlap = poly_iou_list(forward_state['old_target_pos'][::1+time_dilate],forward_state['old_target_sz'][::1+time_dilate],inv_target_pos[::1+time_dilate],inv_target_sz[::1+time_dilate])

        
        
        #forward_state['old_target_pos'] = state_old_pos
        #forward_state['old_target_sz'] = state_old_sz
        #forward_state['old_target_score'] = state_old_score

        #forward_state['inv_target_pos'] = inv_target_pos
        #forward_state['inv_target_sz'] = inv_target_sz
        #forward_state['inv_target_score'] = inv_score
        
        #forward_state['inviou'] = b_overlap
        forward_state['pred_target_pos'] = forward_state['target_pos']
        forward_state['pred_target_sz'] = forward_state['target_sz']
        forward_state['pred_target_score'] = forward_state['score']
        (kf_pred_x,kf_pred_y) = forward_state['KF'].predict()
        kf_pred_pos = np.squeeze(np.asarray([kf_pred_x,kf_pred_y]))[0]
        forward_state['KF_predict_pos'] = kf_pred_pos

        
        #if pass_check_mode:
            #if b_overlap<0.7:
                #forward_state['target_pos'] = (state_old_pos+forward_state['target_pos'])/2                
                #forward_state['target_pos'] = kf_pred_pos
                #forward_state['KF_predict_pos'] = forward_state['target_pos']
                #forward_state['KF_est_pos'] = forward_state['target_pos']
                
                #print(np.squeeze(np.asarray(state['KF_predict_pos']))[0])
                #print(np.squeeze(np.asarray(state['KF_est_pos']))[0])

                
                #forward_state['target_sz'] = (state_old_sz+forward_state['target_sz'])/2
                #forward_state['target_schore'] = 1
                #forward_state['imnow'] = state_old_im
            #else:
                #print(forward_state['target_pos'])
                
                #(kf_est_x,kf_est_y) = forward_state['KF'].update(np.array([forward_state['target_pos'][0],forward_state['target_pos'][1]]))
                #forward_state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                
        #else:
            #(kf_est_x,kf_est_y) = forward_state['KF'].update(np.array([forward_state['target_pos'][0],forward_state['target_pos'][1]]))
            #kf_est_pos = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
            #forward_state['KF_est_pos'] = kf_est_pos
            
            
        #toc_writemem += cv2.getTickCount() - tic
        
        
        #toc += cv2.getTickCount() - tic_total
        
        #toc*=1000
        #toc_track1*=1000
        #toc_inv*=1000
        #toc_writemem*=1000
        
        #toc /=cv2.getTickFrequency()
        #toc_track1/=cv2.getTickFrequency()
        #toc_inv/=cv2.getTickFrequency()
        #toc_writemem/=cv2.getTickFrequency()
        #print('total time track2pass = {:2.1f}, toc_track1 = {:2.1f}, toc_inv = {:2.1f}, toc_writemem = {:2.1f}'.format(toc,toc_track1/toc,toc_inv/toc,toc_writemem/toc))
        
        return forward_state
    
    
    
    def update_need_del_index(self,_use_flag_next,flag,inv_frames):
        use_flag_next = copy.deepcopy(_use_flag_next)
        use_flag_next.append(flag)

        need_del_index = use_flag_next.index(1)
        if sum(np.array(use_flag_next)==1)<inv_frames:
            need_del_index = 0
        use_flag_next = use_flag_next[need_del_index:]

        addlengh = sum(np.array(use_flag_next)==0)
        if len(use_flag_next) > inv_frames+addlengh:
            needdel2 = (len(use_flag_next)-inv_frames-addlengh)

            use_flag_next = use_flag_next[needdel2:]
            need_del_index = need_del_index + needdel2
        if sum(np.array(use_flag_next)==1)>inv_frames:
            dsadsa

        #nextaddlengh = sum(np.array(use_flag_next)==0)
        #print('use_flag next update! now is = ',use_flag_next)
        #print('need_del_index = ',need_del_index)
        
        return use_flag_next,need_del_index
    
    
    def scorematrix2costmatrix(self,scorematrix):
        #change score to cost
        #print('costmatrix ori = ',scorematrix)
        now_max_iou = np.max(scorematrix)
        cost_matrix = []
        for row in scorematrix:
            cost_row = []
            for col in row:
                cost_row += [now_max_iou - col]
            cost_matrix += [cost_row]


        cost_matrix = np.array(cost_matrix)
        #print(cost_matrix)

        if np.size(cost_matrix,0)-np.size(cost_matrix,1)>0:
            cost_matrix = np.pad(cost_matrix,((0,0),(0,np.size(cost_matrix,0)-np.size(cost_matrix,1))),'constant',constant_values=(now_max_iou,now_max_iou))
        if np.size(cost_matrix,0)-np.size(cost_matrix,1)<0:
            cost_matrix = np.pad(cost_matrix,((0,np.size(cost_matrix,1)-np.size(cost_matrix,0)),(0,0)),'constant',constant_values=(now_max_iou,now_max_iou))
        #print('max(scorematrix)- scorematrix = cost_matrix = ',cost_matrix)

        return cost_matrix
    
    
    def inv_neighbor_single(self,im,oldim,location_neighbor_pos,location_neighbor_sz,state\
                            ,now_frame_index,inv_start_from_history,old_target_pos_sz):
        l_pos=[]
        l_sz=[]
        l_posind=[]
        l_score=[]
        for ind in range(len(location_neighbor_pos)):
            #inverse frames from model output not miximize target pos
            if 'online_tracker_inv' in state:
                n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,oldim,np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],online_tracker_inv=state['online_tracker_inv'],dataname=state['dataname'],resume=state['resume'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
            else:
                n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,oldim,np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
            l_pos.append(n_inv_pos.copy())
            l_sz.append(n_inv_sz.copy())
            l_posind.append(n_inv_index.copy())
            l_score.append(n_inv_score.copy())
        return l_pos,l_sz,l_posind,l_score
    
    
    
    def inv_neighbor_group(self,im,oldim,location_neighbor_pos,\
                           location_neighbor_sz,location_neighbor_score,state,\
                           now_frame_index,inv_start_from_history,\
                           old_target_pos_sz,neighbor_score_only,neighbor_static_th,neighbor_th):
        #print('workwork')
        neighbor_states=[]
        _now_frame_index = copy.copy(now_frame_index)
        for ind_inv in range(len(oldim)):
            old_im_step = oldim[-1-ind_inv]
            now_bboxes=[]
            now_pos=[]
            now_sz=[]
            now_score=[]
            for ind_neighbors in range(len(location_neighbor_pos)):
                #inverse frames from model output not miximize target pos
                now_n_pos = np.array(copy.deepcopy(location_neighbor_pos[ind_neighbors]))
                now_n_sz = np.array(copy.deepcopy(location_neighbor_sz[ind_neighbors]))
                now_n_score = np.array(location_neighbor_score[ind_neighbors])
                if ind_inv==0:
                    #step 1 get all neighbors inverse bbox
                    #print('now_n_sz= ',now_n_sz)
                    #print('type(now_n_sz)= ',type(now_n_sz))

                    #print('now_n_pos= ',now_n_pos)
                    #print('type(now_n_pos)= ',type(now_n_pos))
                    state_invobj= self.track_inv_neighbor(im,old_im_step,now_n_pos,now_n_sz,state['net'],state['p'],neighbor_score_only=neighbor_score_only,neighbor_static_th=neighbor_static_th,neighbor_th=neighbor_th)
                    state_invobj['start_pos']=now_n_pos
                    state_invobj['start_sz']=now_n_sz
                    state_invobj['start_score']=now_n_score
                    state_invobj['n_inv_pos']=[]
                    state_invobj['n_inv_sz']=[]
                    state_invobj['n_inv_score']=[]
                    state_invobj['n_inv_posind']=[]

                else:
                    
                    neighbor_states[ind_hung]['next_sz']
                    now_n_pos = copy.deepcopy(neighbor_states[ind_hung]['next_pos'])
                    now_n_sz = copy.deepcopy(neighbor_states[ind_hung]['next_sz'])
                    now_n_pos = np.array(now_n_pos)
                    now_n_sz = np.array(now_n_sz)
                    old_im_step0 = oldim[-1-ind_inv+1]
                    #print('now_n_sz= ',now_n_sz)
                    #print('type(now_n_sz)= ',type(now_n_sz))

                    #print('now_n_pos= ',now_n_pos)
                    #print('type(now_n_pos)= ',type(now_n_pos))

                    state_invobj= self.track_inv_neighbor(old_im_step0,old_im_step,now_n_pos,now_n_sz,state['net'],state['p'],state=neighbor_states[ind_hung],neighbor_score_only=neighbor_score_only,neighbor_static_th=neighbor_static_th,neighbor_th=neighbor_th)




                n_pos=[]
                n_sz=[]
                n_bbox_score=[]

                #add target maxima to neighbor
                #state_invobj['n_target_pos'][0].append(copy.deepcopy(state_invobj['target_pos'][0]))
                #state_invobj['n_target_pos'][1].append(copy.deepcopy(state_invobj['target_pos'][1]))
                #state_invobj['n_target_sz'][0].append(copy.deepcopy(state_invobj['target_sz'][0]))
                #state_invobj['n_target_sz'][1].append(copy.deepcopy(state_invobj['target_sz'][1]))
                #print(state_invobj['n_score'])
                #print(np.append(state_invobj['n_score'],copy.deepcopy([state_invobj['score']]),axis=0))

                state_invobj['n_score']=np.append(state_invobj['n_score'],\
                                  copy.deepcopy([state_invobj['score']]),axis=0)

                for ind_n_target_p in range(state_invobj['n_target_pos'][0].size):


                    rectn_ = cxy_wh_2_rect([state_invobj['n_target_pos'][0][ind_n_target_p],state_invobj['n_target_pos'][1][ind_n_target_p]], [state_invobj['n_target_sz'][0][ind_n_target_p],state_invobj['n_target_sz'][1][ind_n_target_p]])
                    n_bbox_score.append([rectn_[0],rectn_[1],rectn_[0]+rectn_[2], \
                                                       rectn_[1]+rectn_[3],state_invobj['n_score'][ind_n_target_p]])

                    n_pos.append([state_invobj['n_target_pos'][0][ind_n_target_p],\
                                  state_invobj['n_target_pos'][1][ind_n_target_p]])
                    n_sz.append([state_invobj['n_target_sz'][0][ind_n_target_p],\
                                 state_invobj['n_target_sz'][1][ind_n_target_p]])

                    now_pos.append([state_invobj['n_target_pos'][0][ind_n_target_p],\
                                  state_invobj['n_target_pos'][1][ind_n_target_p]])
                    now_sz.append([state_invobj['n_target_sz'][0][ind_n_target_p],\
                                   state_invobj['n_target_sz'][1][ind_n_target_p]])
                    now_score.append(state_invobj['n_score'][ind_n_target_p])


                    now_bboxes.append([rectn_[0],rectn_[1],rectn_[0]+rectn_[2],\
                                       rectn_[1]+rectn_[3],state_invobj['n_score'][ind_n_target_p]])

                #add target maxima to neighbor
                rectn_ = cxy_wh_2_rect(state_invobj['target_pos'],state_invobj['target_sz'])
                now_bboxes.append([rectn_[0],rectn_[1],rectn_[0]+rectn_[2],\
                                       rectn_[1]+rectn_[3],copy.deepcopy(state_invobj['score'])])
                n_pos.append(state_invobj['target_pos'])
                n_sz.append(state_invobj['target_sz'])
                now_pos.append(state_invobj['target_pos'])
                now_sz.append(state_invobj['target_sz'])
                now_score.append(state_invobj['score'])

                state_invobj['tn_bbox_score']=n_bbox_score
                state_invobj['n_pos']=n_pos
                state_invobj['n_sz']=n_sz
                if ind_inv==0:
                    neighbor_states.append(state_invobj)
            all_iou_score_max=[]
            _now_bboxes = np.array(copy.deepcopy(now_bboxes))
            _now_bboxes = torch.tensor(_now_bboxes)
            #print('nms_sigma = ',nms_sigma)
            #print('nms_th = ',nms_th)
            n_nms_bbox_scores,n_nms_bbox_index = softnms_cpu_torch(_now_bboxes,score_threshold=0.01,output_index=1,sigma=0.5)
            #print('len(_now_bboxes) = ',len(_now_bboxes))
            #print('len(n_nms_bbox_scores) = ',len(n_nms_bbox_scores))
            #print(n_nms_bbox_scores)
            now_pos = [now_pos[ind] for ind in n_nms_bbox_index]
            now_sz = [now_sz[ind] for ind in n_nms_bbox_index]
            now_score = [now_score[ind] for ind in n_nms_bbox_index]
            now_bboxes = [now_bboxes[ind] for ind in n_nms_bbox_index]

            for ind_neighbors in range(len(location_neighbor_pos)):
                polys2pos = neighbor_states[ind_neighbors]['n_pos']
                polys2sz = neighbor_states[ind_neighbors]['n_sz']
                #print('len(now_pos) = ',len(now_pos))
                #print('len(neighbor_states) = ',len(neighbor_states))
                #print('len(location_neighbor_pos) = ',len(location_neighbor_pos))
                #print('len(now_bboxes) = ',len(now_bboxes))

                iou_matrix=poly_iou_list_matrix(now_pos,now_sz, polys2pos,polys2sz)
                iou_matrix=iou_matrix*\
                np.expand_dims(neighbor_states[ind_neighbors]['n_score'],0).repeat(len(now_pos),axis=0)
                iou_matrix = iou_matrix.max(1)
                #print('len(score) = ',len(neighbor_states[ind_neighbors]['n_score']))
                #print('len(iou_matrix) = ',np.size(iou_matrix))
                #print(iou_matrix)
                all_iou_score_max.append(iou_matrix)
            all_iou_score_max = np.array(all_iou_score_max)
            #all_iou_score_max[0][0]=0
            #all_iou_score_max[1][1]=0
            #print('bf all_iou_score_max = ',all_iou_score_max)


            cost_matrix = self.scorematrix2costmatrix(all_iou_score_max)

            Mun = Munkres()
            Munindexes = Mun.compute(cost_matrix)
            #print_matrix(Munindexes, msg='Lowest cost through this matrix:')
            #print(Munindexes)

            minusmatrix = copy.deepcopy(all_iou_score_max)*-1
            #print(costmatrix)
            #print(minusmatrix)
            #print(len(all_iou_score_max))
            for ind_hung in range(len(all_iou_score_max)):
                row, column = Munindexes[ind_hung]

                #print('*2  max(costmatrix)- costmatrix = cost_matrix = ',cost_matrix)
                #print('Munindexes = ',Munindexes)
                try:
                    value = all_iou_score_max[row][column]
                    #print('value = ',value)
                    minusmatrix[row][column] = value
                except:
                    pass
                #print('minusmatrix = ',minusmatrix)
            #print('af all_iou_score_max = ',minusmatrix)
            minusmatrix = np.array(minusmatrix)
            for ind_hung in range(len(all_iou_score_max)):
                all_inv_iou = minusmatrix[ind_hung,:]
                #print('all_inv_iou = ',all_inv_iou)
                

                neighbor_states[ind_hung]['n_inv_pos'].insert(0,copy.deepcopy(now_pos[all_inv_iou.argmax()]))
                neighbor_states[ind_hung]['n_inv_sz'].insert(0,copy.deepcopy(now_sz[all_inv_iou.argmax()]))
                neighbor_states[ind_hung]['n_inv_score'].insert(0,copy.deepcopy(now_score[all_inv_iou.argmax()]))
                neighbor_states[ind_hung]['n_inv_posind'].insert(0,_now_frame_index-ind_inv-1)
                neighbor_states[ind_hung]['next_pos']=now_pos[all_inv_iou.argmax()]
                neighbor_states[ind_hung]['next_sz']=now_sz[all_inv_iou.argmax()]

        l_pos=[]
        l_sz=[]
        l_score=[]
        l_posind=[]
        for ind_l in range(len(location_neighbor_pos)):
            l_pos.append(neighbor_states[ind_l]['n_inv_pos'])
            l_sz.append(neighbor_states[ind_l]['n_inv_sz'])
            l_score.append(neighbor_states[ind_l]['n_inv_score'])
            l_posind.append(neighbor_states[ind_l]['n_inv_posind'])
        return l_pos,l_sz,l_posind,l_score


    
    def _neighbor_track(self, state, im, warning_inv_th=0,inv_error_th=0,gtchoose=False,just_kalman=False,neighbor_2pass_mode=0,neighbor_predandinv_mode=0,one_target_pred_first=0,kalman_add_mode=0,ls_add_mode=0,inv_frames=1,ignore_frames=0,kalman_neighbor_mode=0,time_dilate=0,online_score=None, gt=None,pass_check_mode=False,dynamic_ig=None,now_frame_index=0,nms_th=0.25,nms_sigma=0.01,delay_start=False,inv_start_from_history=False,history_dict=None,max_consecutive=-1,pred_margin_ratio=1.0,neighbor_search_ratio=1.0,neighbor_score_only=False,neighbor_static_th=False,neighbor_th=0.7,group_inv=False):
        #print('now_frame_index = ',now_frame_index)
        
        
        #----------------------------- flag setting ---------------------------------
        if delay_start:
            state['delay_start']=True
        else:
            state['delay_start']=False
        state['neighbor_score_only'] = neighbor_score_only
        state['neighbor_static_th'] = neighbor_static_th
        state['neighbor_th'] = neighbor_th
        #print('state[neighbor_score_only] = ',state['neighbor_score_only'])
        state['neighbor_search_ratio'] = neighbor_search_ratio

        warning_2pass = state['warning_2pass']
        max_iou_in_model_pred = state['max_iou_in_model_pred']
        max_iou_in_kalman_pred = state['max_iou_in_kalman_pred']
        max_iou_in_neighbor_pred = state['max_iou_in_neighbor_pred']
        choose_right_pred = state['choose_right_pred']
        
        if warning_inv_th=='dy':
            _warning_inv_th = state['CCmean']-state['CCstd']
            print('now CCmean = {}, CCstd = {}, 1std TH = {}, history={}'.format(state['CCmean'],state['CCstd'],_warning_inv_th,state['historyCCinv']))
            if _warning_inv_th<0:
                dsadsadas
        else:
            _warning_inv_th = warning_inv_th
        if dynamic_ig =='warninginv':
            dynamic_ig = 1
        if dynamic_ig =='KF':
            dynamic_ig = 2
            
        if dynamic_ig:
            if 'use_flag_next' in state:
                state['use_flag'] = copy.deepcopy(state['use_flag_next'])
                #print('use_flag update! now is = ',state['use_flag'])


        KF_index = -1
        #warning_inv_th = np.float64(args.th_warning_inv)
        #inv_error_th = np.float64(args.th_KF) #less than =>KF
        #gtchoose=0 # use gt to choose final answer
        #just_kalman=0 #always use kalman to choose final answer
        #neighbor_2pass_mode=np.int64(args.neighbor) #use t-1 neighbor and t neighbor matching
        #neighbor_predandinv_mode=np.int64(args.neighbor_predandinv) # neighbor iou = IOU(pred,targetbox) + IOU(neighbor,inv(targetbox))
        #one_target_pred_first=np.int64(args.trust_one_target) # if just one target in predict, choose it
        #kalman_add_mode = np.int64(args.KF_add_neighbor) #if KF ,add pred to neighbor

        
        #tic = cv2.getTickCount()
        #toc += cv2.getTickCount() - tic
        #toc /= cv2.getTickFrequency()
        
        
        
        #  track pred 
        state = self.track_2pass(state, im,online_score=online_score,pass_check_mode = False,inv_frames=inv_frames,time_dilate=time_dilate,now_frame_index=now_frame_index)
        
        if inv_start_from_history:
            if 'old_target_pos' in state:
                old_target_pos_sz=[state['old_target_pos'],state['old_target_sz']]
            else:
                old_target_pos_sz=None
        else:
            old_target_pos_sz=None


        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        #location_old = cxy_wh_2_rect(state['old_target_pos'], state['old_target_sz'])
        location_pred = cxy_wh_2_rect(state['pred_target_pos'], state['pred_target_sz'])
        if 'n_target_pos' in state:
            #print('n_target_pos = ',state['n_target_pos'])
            location_neighbor_pred = []
            location_neighbor_pos = []
            location_neighbor_sz = []
            location_neighbor_posind=[]
            location_neighbor_score=[]
            location_neighbor_xyxy = []
            for ind in range(state['n_target_pos'][0].size):
                rectind = cxy_wh_2_rect([state['n_target_pos'][0][ind],state['n_target_pos'][1][ind]], [state['n_target_sz'][0][ind],state['n_target_sz'][1][ind]])
                location_neighbor_pred.append(rectind)

                location_neighbor_pos.append([state['n_target_pos'][0][ind],state['n_target_pos'][1][ind]])
                location_neighbor_sz.append([state['n_target_sz'][0][ind],state['n_target_sz'][1][ind]])
                location_neighbor_posind.append(now_frame_index)
                location_neighbor_score.append(state['n_score'])
                location_neighbor_xyxy.append([rectind[0],rectind[1],rectind[0]+rectind[2], \
                                                   rectind[1]+rectind[3],state['n_score'][ind]])

            location_neighbor_xyxy.append([location_pred[0],location_pred[1],location_pred[0]+location_pred[2], \
                                               location_pred[1]+location_pred[3],1.01])

            location_neighbor_xyxy = torch.tensor(location_neighbor_xyxy)
            #print('nms_th = ',nms_th)                
            nms_bbox_scores,nms_bbox_index = softnms_cpu_torch(location_neighbor_xyxy,score_threshold=nms_th,output_index=1,sigma=nms_sigma)
            #nms_bbox_scores,nms_bbox_index = softnms_cpu_torch(location_neighbor_xyxy,score_threshold=0.0,output_index=1,sigma=0.01)
            #print('nms_bbox_scores = ',nms_bbox_scores)
            #print('nms_bbox_index = ',nms_bbox_index)

            if max(nms_bbox_index) == len(location_neighbor_xyxy)-1:
                argmaxnmsbbox = np.argmax(nms_bbox_index)
                nms_bbox_scores[argmaxnmsbbox]=[]
                nms_bbox_index[argmaxnmsbbox]=[]
                while [] in nms_bbox_scores:
                    nms_bbox_scores.remove([])
                    nms_bbox_index.remove([])
            else:
                print('nms_bbox_scores = ',nms_bbox_scores)
                print('nms_bbox_index = ',nms_bbox_index)
                dasdsadsa



            #print('nms_bbox_scores = ',nms_bbox_scores)
            #print('nms_bbox_index = ',nms_bbox_index)

            location_neighbor_pred = [location_neighbor_pred[ind] for ind in nms_bbox_index]
            location_neighbor_pos = [location_neighbor_pos[ind] for ind in nms_bbox_index]                
            location_neighbor_posind = [location_neighbor_posind[ind] for ind in nms_bbox_index]                
            location_neighbor_score = [location_neighbor_score[ind] for ind in nms_bbox_index]                
            location_neighbor_sz = [location_neighbor_sz[ind] for ind in nms_bbox_index]
            #print('location_neighbor_pos = ',location_neighbor_pos)


            #location_neighbor_pos = np.array(location_neighbor_pos)
            #location_neighbor_sz = np.array(location_neighbor_sz)
        #print('state[online_tracker_inv] = ',state['online_tracker_inv'])
        
        if len(location_neighbor_pos)+len(state['old_neighbor_pos'])>0 and inv_frames!=0:
            
            state_old_im = state['oldim'].copy()
            # inverse frames from model output miximize target pos
                
            if 'online_tracker_inv' in state:
                inv_target_pos, inv_target_sz, inv_score,inv_index = self.track_inv(im,state_old_im,state['target_pos'], state['target_sz'], state['net'], state['p'],online_score=online_score,gt=gt,online_tracker_inv=state['online_tracker_inv'],dataname=state['dataname'],resume=state['resume'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
            else:
                inv_target_pos, inv_target_sz, inv_score,inv_index = self.track_inv(im,state_old_im,state['target_pos'], state['target_sz'], state['net'], state['p'],online_score,gt,now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
            
            if dynamic_ig and 'use_flag' in state:
                try:
                    if not history_dict is None:
                        eachiou = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1)[1]
                        if not gt is None:
                            eachiouHistGT = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1,poly2posisgt=1)[1]
                            eachiouGT = poly_iou_list(inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1,poly2posisgt=1)[1]

                        
                    b_overlap = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:],useflag=state['use_flag'])

                except:
                    print('state[old_target_posind][::1+time_dilate][ignore_frames:] = ',state['old_target_posind'][::1+time_dilate][ignore_frames:])
                    print('inv_index[::1+time_dilate][ignore_frames:] = ',inv_index[::1+time_dilate][ignore_frames:])
                    print('state[use_flag] = ',state['use_flag'])
                    print(state['old_target_pos'][::1+time_dilate][ignore_frames:])
                    print(inv_target_pos[::1+time_dilate][ignore_frames:])

                    dsadsada                    


                
            else:
                if not history_dict is None:
                    eachiou = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:],outputeachiou=1)[1]
                    if not gt is None:
                        eachiouHistGT = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],outputeachiou=1,poly2posisgt=1)[1]
                        eachiouGT = poly_iou_list(inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],outputeachiou=1,poly2posisgt=1)[1]


                
                
                
                b_overlap = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:])

            
            state['inv_target_pos'] = inv_target_pos
            state['inv_target_sz'] = inv_target_sz
            state['inv_target_score'] = inv_score
            state['inv_target_posind'] = inv_index
            #print('state[inv_target_pos] = ',state['inv_target_pos'])
            #print('state[inv_target_posind] = ',state['inv_target_posind'])
            
            state['inviou'] = b_overlap
            state['inviou_ori'] = b_overlap

            
        else:
            state['inv_target_pos'] = []
            state['inv_target_sz'] = []
            state['inv_target_score'] = []
            state['inv_target_posind'] = []


            #state_old_im = state['oldim'].copy()

            #inv_target_pos, inv_target_sz, inv_score = self.track_inv(im,state_old_im,state['target_pos'], state['target_sz'], state['net'], state['p'],online_score,gt)
            
            #b_overlap = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],inv_target_pos[::1+time_dilate][ignore_frames:],inv_target_sz[::1+time_dilate][ignore_frames:])
            
            #state['inv_target_pos'] = inv_target_pos
            #state['inv_target_sz'] = inv_target_sz
            #state['inv_target_score'] = inv_score
            #state['inviou_ori'] = b_overlap

            
            
            state['inviou'] = 1.01
            
            
        nextaddlengh=0
        if warning_inv_th=='dy':
            if state['inviou']<1.01:
                state['historyCCinv'].append(state['inviou'])
                state['CCmean']=np.mean(np.array(state['historyCCinv']))
                state['CCstd']=np.std(np.array(state['historyCCinv']),ddof=1)

        if dynamic_ig:
            
            
            
            if not 'use_flag' in state:
                state['use_flag_next'] = [1]
                state['need_del_index_next'] = 0
                
                
            state['use_flag'] = copy.deepcopy(state['use_flag_next'])
            state['need_del_index'] = copy.deepcopy(state['need_del_index_next'])

            
            if dynamic_ig==1:
                if state['inviou'] < _warning_inv_th:
                    _flag=0
                else:
                    _flag=1


                (state['use_flag_next'],state['need_del_index_next']) = self.update_need_del_index(state['use_flag'],_flag,inv_frames)


            


            
        location_KF_pred = cxy_wh_2_rect(state['KF_predict_pos'], state['old_target_sz'][-1])
        
        if not gt is None:
            b_overlap = poly_iou(gt, location_pred)
        
        
        
        
        
        

        


        all_inv_iou=[]
        state['choose']='-1'
        if max_consecutive==-1:
            consecutive_flag=1
        else:
            if state['nowconsecutive']>max_consecutive:
                consecutive_flag=0
            else:
                consecutive_flag=1
        if state['inviou'] < _warning_inv_th and state['delay_start_flag']==1 and consecutive_flag:
            warning_2pass += 1
            state['flag_max_iou_in_model_pred']=0
            state['flag_max_iou_after_method']=0
            state['flag_output_image']=0
            state['nowconsecutive']=state['nowconsecutive']+1
            if not history_dict is None:
                #max part history
                history_dict['nowconsecutive'] = copy.copy(state['nowconsecutive'])
                history_dict[history_dict['nowname']]['frame'].append(now_frame_index)                        
                history_dict[history_dict['nowname']]['pos_sz'].append([state['inv_target_pos'],state['inv_target_sz']])
                history_dict[history_dict['nowname']]['score'].append(state['inv_target_score'])
                history_dict[history_dict['nowname']]['invweight'].append(copy.copy(eachiou))
                if not gt is None:
                    history_dict[history_dict['nowname']]['invweightGT'].append(copy.copy(eachiouGT))
                    history_dict[history_dict['nowname']]['invweightHistGT'].append(copy.copy(eachiouHistGT))
                    
                history_dict[history_dict['nowname']]['consecutive'].append(copy.copy(history_dict['nowconsecutive']))
                history_dict[history_dict['nowname']]['GTIOU'].append(b_overlap)
                history_dict[history_dict['nowname']]['nowinviou'].append(state['inviou'])


            
            
            if just_kalman:
                max_iou_in_kalman_pred += 1
                state['target_pos'] = state['KF_predict_pos']
                state['target_sz'] = state['old_target_sz'][-1]
                state['score'] = np.float64(1)
                b_overlap_KFpred = poly_iou(gt, location_KF_pred)
                b_overlap = b_overlap_KFpred
                state['KF_est_pos'] = state['target_pos']
                state['choose']='KF0'
            else:
                if not gt is None:
                    #choose by gt
                    
                    b_overlap_KFpred = poly_iou(gt, location_KF_pred)
                    b_overlap_npred = [poly_iou(gt, n_pred) for n_pred in location_neighbor_pred]
                    all_overlap = []
                    all_overlap.append(b_overlap)
                    all_overlap.append(b_overlap_KFpred)
                    for b_o_n in b_overlap_npred:
                        all_overlap.append(b_o_n)
                    all_overlap = np.array(all_overlap)
                    #print('all_overlap.argmax() = ',all_overlap.argmax())


                    if all_overlap.argmax()==0:
                        max_iou_in_model_pred += 1
                        state['flag_max_iou_in_model_pred']=1

                        if gtchoose:
                            (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                            state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]

                    if all_overlap.argmax()==1:
                        max_iou_in_kalman_pred += 1
                        if gtchoose:

                            state['target_pos'] = state['KF_predict_pos']
                            state['target_sz'] = state['old_target_sz'][-1]
                            state['score'] = np.float64(1)
                            #b_overlap = all_overlap.max()
                            state['KF_est_pos'] = state['target_pos']
                            

                    if all_overlap.argmax()>1:
                        max_iou_in_neighbor_pred += 1
                        if gtchoose:
                            now_max_iou_ind = nms_bbox_index[np.array(all_overlap).argmax()-2]

                            state['target_pos'][0] = state['n_target_pos'][0][now_max_iou_ind]
                            state['target_pos'][1] = state['n_target_pos'][1][now_max_iou_ind]
                            state['target_sz'][0] = state['n_target_sz'][0][now_max_iou_ind]
                            state['target_sz'][1] = state['n_target_sz'][1][now_max_iou_ind]
                            state['score'] = state['n_score'][now_max_iou_ind]
                            (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                            state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]

                            #b_overlap = all_overlap.max()

                if gtchoose==0:
                    
                    
                    if kalman_neighbor_mode:
                        location_neighbor_pos.append(state['KF_predict_pos'])
                        location_neighbor_sz.append(state['old_target_sz'][-1])
                        location_neighbor_posind.append(now_frame_index)
                        location_neighbor_score.append(0)
                        KF_index = len(location_neighbor_pos)
                        print(len(location_neighbor_pos))
                        print('KF in location_neighbor_pos')

                    costmatrix=np.zeros((len(state['old_neighbor_pos'])+1,len(location_neighbor_pos)+1))

                    #print('costmatrix= ',costmatrix)
                    #print('old_neighbor_pos = ',state['old_neighbor_pos'])
                    #print('location_neighbor_pos = ',location_neighbor_pos)

                    if len(np.array(location_neighbor_pos).flatten())==1 and one_target_pred_first:# if just one target then choose it
                        (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                        state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                        state['choose']='model pred0'

                    else:
                        
                        
                        n_inv_pred_rect=[]
                        n_inv_pred_pos=[]
                        n_inv_pred_sz=[]
                        n_inv_pred_posind=[]
                        n_inv_pred_score=[]

                        
                        
                        #location_olds = []
                        #for ind in range(len(state['old_target_pos'])):
                        #    location_old = cxy_wh_2_rect(state['old_target_pos'][-1-ind], state['old_target_sz'][-1-ind])
                        #    location_old = np.array(location_old)
                        #    location_olds.append(location_old)

                        
                        if not group_inv:
                            l_pos_,l_sz_,l_posind_,l_score_=\
                            self.inv_neighbor_single(im,state['oldim'],location_neighbor_pos,location_neighbor_sz,\
                                                     state,now_frame_index,inv_start_from_history,old_target_pos_sz)
                        else:
                            location_neighbor_pos.insert(0,copy.deepcopy(state['target_pos']))
                            location_neighbor_sz.insert(0,copy.deepcopy(state['target_sz']))
                            location_neighbor_posind.insert(0,now_frame_index)
                            location_neighbor_score.insert(0,copy.deepcopy(state['score']))

                            
                            l_pos_,l_sz_,l_posind_,l_score_=\
                            self.inv_neighbor_group(im,state['oldim'],location_neighbor_pos,location_neighbor_sz,location_neighbor_score,\
                                                     state,now_frame_index,inv_start_from_history,old_target_pos_sz,neighbor_score_only,neighbor_static_th,neighbor_th)

                            
                        #print('len(location_neighbor_pos) = ',len(location_neighbor_pos))
                        for ind in range(len(location_neighbor_pos)):
                            #inverse frames from model output not miximize target pos

                            '''
                            if 'online_tracker_inv' in state:
                                n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,state['oldim'],np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],online_tracker_inv=state['online_tracker_inv'],dataname=state['dataname'],resume=state['resume'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
                            else:
                                n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,state['oldim'],np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
                            '''


                            #print('location_neighbor_pos[ind]',location_neighbor_pos[ind])
                            #print('location_neighbor_sz[ind]',location_neighbor_sz[ind])
                            #print('state[target_pos] = ',state['target_pos'])
                            #print('state[target_sz] = ',state['target_sz'])



                            #l_pos = n_inv_pos.copy()
                            #l_sz = n_inv_sz.copy()
                            #l_posind = n_inv_index.copy()
                            #l_score = n_inv_score.copy()

                            #print('l_pos = ',l_pos)
                            #print('l_pos_[0] = ',l_pos_[0])
                            #dsadsadsa
                            #n_inv_pos = l_pos[ind].copy()
                            #n_inv_sz = l_sz[ind].copy()
                            n_inv_pred_pos.append(l_pos_[ind].copy())
                            n_inv_pred_sz.append(l_sz_[ind].copy())
                            n_inv_pred_posind.append(l_posind_[ind].copy())
                            n_inv_pred_score.append(l_score_[ind].copy())

                            l_pos_[ind].append(np.array(location_neighbor_pos[ind]))
                            l_sz_[ind].append(np.array(location_neighbor_sz[ind]))
                            l_posind_[ind].append(np.array(location_neighbor_posind[ind]))
                            l_score_[ind].append(np.array(location_neighbor_score[ind]))



                            #print('n_inv_index = ',n_inv_index)
                            #print('np.array(location_neighbor_posind[ind]) = ',np.array(location_neighbor_posind[ind]))

                            #print('now frame  = ',now_frame_index)
                            #print('len(oldim) = ',len(state['oldim']))
                            #print('l_posv4 = ',l_pos)
                            location_neighbor_pos[ind] = l_pos_[ind] # now+inv
                            location_neighbor_sz[ind] = l_sz_[ind]
                            location_neighbor_posind[ind] = l_posind_[ind]
                            location_neighbor_score[ind] = l_score_[ind]

                            '''
                            rect_ns = []
                            for ind2 in range(len(n_inv_pos)):
                                rectn = cxy_wh_2_rect(n_inv_pos[-1-ind2], n_inv_sz[-1-ind2])
                                rectn = np.array(rectn)
                                rect_ns.append(rectn)
                            n_inv_pred_rect.append(rect_ns)
                            '''

                        if not group_inv:
                            all_inv_iou.append(state['inviou'])
                        print('---------------now in pred inviou---------------------')
                        print('all_inv_iou = ',all_inv_iou)
                        print('state[inviou] = ',state['inviou'])
                        print('pred_margin_ratio = ',pred_margin_ratio)
                        print('state[inviou] = ',state['inviou']*pred_margin_ratio)
                        print('------------------------------------------------------')

                            
                        '''
                        location_neighbor_xyxy = torch.tensor(location_neighbor_xyxy)
                        #print('nms_th = ',nms_th)                
                        nms_bbox_scores,nms_bbox_index = softnms_cpu_torch(location_neighbor_xyxy,score_threshold=nms_th,output_index=1,sigma=nms_sigma)



                        def poly_iou_list(polys1pos,polys1sz, polys2pos,polys2sz, bound=None,useflag=None,outputeachiou=None,poly2posisgt=None):
                        '''
                                    
                                    





                        
                        if not dynamic_ig:
                            b_overlap_npred_inv = [poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:]) for ind in range(len(n_inv_pred_pos))]
                            
                            if not history_dict is None:
                                #neighbor part history

                                for ind in range(len(n_inv_pred_pos)):
                                    eachiou = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:],outputeachiou=1)[1]
                                    if not gt is None:
                                        eachiouHistGT = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],outputeachiou=1,poly2posisgt=1)[1]
                                        eachiouGT = poly_iou_list(n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],outputeachiou=1,poly2posisgt=1)[1]
                                        history_dict[history_dict['nowname']]['invweightGT'].append(copy.copy(eachiouGT))
                                        history_dict[history_dict['nowname']]['invweightHistGT'].append(copy.copy(eachiouHistGT))

                                    
                                    history_dict[history_dict['nowname']]['frame'].append(now_frame_index)
                                    history_dict[history_dict['nowname']]['pos_sz'].append([n_inv_pred_pos[ind],n_inv_pred_sz[ind]])
                                    history_dict[history_dict['nowname']]['invweight'].append(copy.copy(eachiou))
                                    history_dict[history_dict['nowname']]['consecutive'].append(copy.copy(history_dict['nowconsecutive']))
                                    rectnnow = cxy_wh_2_rect(location_neighbor_pos[ind][-1], location_neighbor_sz[ind][-1])
                                    b_overlap_ngt = poly_iou(gt, rectnnow)
                                    history_dict[history_dict['nowname']]['GTIOU'].append(b_overlap_ngt)
                                    history_dict[history_dict['nowname']]['nowinviou'].append(b_overlap_npred_inv[ind])
                                    history_dict[history_dict['nowname']]['score'].append(n_inv_pred_score[ind])
                                    
                    
                            
                        else:
                            try:

                                
                                
                                
                                
                                
                                
                                b_overlap_npred_inv = [poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:],useflag=state['use_flag']) for ind in range(len(n_inv_pred_pos))]
                                
                                if not history_dict is None:
                                    #neighbor part history

                                    for ind in range(len(n_inv_pred_pos)):
                                        eachiou = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1)[1]
                                        if not gt is None:
                                            eachiouHistGT = poly_iou_list(state['old_target_pos'][::1+time_dilate][ignore_frames:],state['old_target_sz'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1,poly2posisgt=1)[1]
                                            eachiouGT = poly_iou_list(n_inv_pred_pos[ind][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],state['old_gt'][::1+time_dilate][ignore_frames:],useflag=state['use_flag'],outputeachiou=1,poly2posisgt=1)[1]
                                            history_dict[history_dict['nowname']]['invweightGT'].append(copy.copy(eachiouGT))
                                            history_dict[history_dict['nowname']]['invweightHistGT'].append(copy.copy(eachiouHistGT))


                                        history_dict[history_dict['nowname']]['frame'].append(now_frame_index)
                                        history_dict[history_dict['nowname']]['pos_sz'].append([n_inv_pred_pos[ind],n_inv_pred_sz[ind]])
                                        history_dict[history_dict['nowname']]['invweight'].append(copy.copy(eachiou))
                                        history_dict[history_dict['nowname']]['consecutive'].append(copy.copy(history_dict['nowconsecutive']))

                                        rectnnow = cxy_wh_2_rect(n_inv_pred_pos[ind], n_inv_pred_sz[ind])
                                        b_overlap_ngt = poly_iou(gt, rectnnow)
                                        history_dict[history_dict['nowname']]['GTIOU'].append(b_overlap_ngt)
                                        history_dict[history_dict['nowname']]['nowinviou'].append(b_overlap_npred_inv[ind])
                                        history_dict[history_dict['nowname']]['score'].append(n_inv_pred_score[ind])


                            except:
                                print('state[old_target_posind][::1+time_dilate][ignore_frames:] = ',state['old_target_posind'][::1+time_dilate][ignore_frames:])
                                print('n_inv_pred_posind[ind][::1+time_dilate][ignore_frames:] = ',n_inv_pred_posind[ind][::1+time_dilate][ignore_frames:])
                                dasdasdsadsa


                        for b_o_n in b_overlap_npred_inv:
                            all_inv_iou.append(b_o_n)
                        print('------------now add neighbor inviou------------------------')
                        print('all_inv_iou = ',all_inv_iou)
                        print('all_inv_iou[0] = ',all_inv_iou[0])
                        print('pred_margin_ratio = ',pred_margin_ratio)
                        print('all_inv_iou[0] = ',all_inv_iou[0]*pred_margin_ratio)
                        print('-----------------------------------------------------------')

                        all_inv_iou[0]=all_inv_iou[0]*pred_margin_ratio






                        if neighbor_2pass_mode:
                            costmatrix[0,]=all_inv_iou
                            _n_inv_pred_pos = []
                            _n_inv_pred_sz = []
                            _n_inv_pred_posind = []

                            if not group_inv:
                                _n_inv_pred_pos.append(state['inv_target_pos'])
                                _n_inv_pred_sz.append(state['inv_target_sz'])
                                _n_inv_pred_posind.append(state['inv_target_posind'])
                                
                            for ind in range(len(n_inv_pred_pos)):
                                _n_inv_pred_pos.append(n_inv_pred_pos[ind])
                                _n_inv_pred_sz.append(n_inv_pred_sz[ind])
                                _n_inv_pred_posind.append(n_inv_pred_posind[ind])
                                
                            n_inv_pred_pos = _n_inv_pred_pos
                            n_inv_pred_sz = _n_inv_pred_sz
                            n_inv_pred_posind = _n_inv_pred_posind
                            
                            #n_inv_pred_pos = [_n_inv_pred_pos.append(x) for x in n_inv_pred_pos]
                            #n_inv_pred_sz = [_n_inv_pred_sz.append(x) for x in n_inv_pred_sz]
                            #n_inv_pred_pos.insert(0,state['inv_target_pos'])
                            #n_inv_pred_sz.insert(0,state['inv_target_sz'])
                            #_location_neighbor_pred = location_neighbor_pred.copy()
                            #_location_neighbor_pred.insert(0,location_pred)
                            
                            if dynamic_ig:
                                state['old_neighbor_pos'] = [x[state['need_del_index']:] for x in state['old_neighbor_pos']]
                                state['old_neighbor_sz'] = [x[state['need_del_index']:] for x in state['old_neighbor_sz']]
                                state['old_neighbor_posind'] = [x[state['need_del_index']:] for x in state['old_neighbor_posind']]
                                
                            else:
                                state['old_neighbor_pos'] = [x[1:] for x in state['old_neighbor_pos'] if len(x)>inv_frames]
                                state['old_neighbor_sz'] = [x[1:] for x in state['old_neighbor_sz'] if len(x)>inv_frames]
                                state['old_neighbor_posind'] = [x[1:] for x in state['old_neighbor_posind'] if len(x)>inv_frames]


                            for ind in range(len(state['old_neighbor_pos'])):

                                if not dynamic_ig:
                                    try:
                                        
                                        b_overlap_nold_inv = [poly_iou_list(state['old_neighbor_pos'][ind][::1+time_dilate][ignore_frames:],state['old_neighbor_sz'][ind][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind2][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind2][::1+time_dilate][ignore_frames:]) for ind2 in range(len(n_inv_pred_pos))]
                                    except:
                                        print('errv1 = ',state['old_neighbor_pos'][ind][::1+time_dilate][ignore_frames:])
                                        print('state[old_neighbor_posind][ind][::1+time_dilate][ignore_frames:] = ',state['old_neighbor_posind'][ind][::1+time_dilate][ignore_frames:])
                                
                                        for ind2 in range(len(n_inv_pred_pos)):
                                            print('n_inv_pred_posind[ind2][::1+time_dilate][ignore_frames:] = ',n_inv_pred_posind[ind2][::1+time_dilate][ignore_frames:])
                                        dasdasdas

                                            
                                else:
                                    #print('neighbor t with t+1')
                                    #print('state[old_neighbor_pos][ind] = ',state['old_neighbor_pos'][ind])
                                    #print('n_inv_pred_pos = ',n_inv_pred_pos)
                                    #print('state[use_flag] = ',state['use_flag'])
                                    try:
                                        b_overlap_nold_inv = [poly_iou_list(state['old_neighbor_pos'][ind][::1+time_dilate][ignore_frames:],state['old_neighbor_sz'][ind][::1+time_dilate][ignore_frames:],n_inv_pred_pos[ind2][::1+time_dilate][ignore_frames:],n_inv_pred_sz[ind2][::1+time_dilate][ignore_frames:],useflag=state['use_flag']) for ind2 in range(len(n_inv_pred_pos))]
                                    except:
                                        print('errv1 = ',state['old_neighbor_pos'][ind][::1+time_dilate][ignore_frames:])
                                        print('state[old_neighbor_posind][ind][::1+time_dilate][ignore_frames:] = ',state['old_neighbor_posind'][ind][::1+time_dilate][ignore_frames:])
                                
                                        for ind2 in range(len(n_inv_pred_pos)):
                                            print('n_inv_pred_posind[ind2][::1+time_dilate][ignore_frames:] = ',n_inv_pred_posind[ind2][::1+time_dilate][ignore_frames:])
                                        dsadsadsad


                                        
                                #print('b_overlap_nold_inv= ',b_overlap_nold_inv)

                                if neighbor_predandinv_mode:
                                    print('not support now')
                                    '''
                                    n_inv_pos, n_inv_sz, n_inv_score = self.track_inv(state['oldim'],im,state['old_neighbor_pos'][ind],state['old_neighbor_sz'][ind],state['net'],state['p'])

                                    rectnpred = cxy_wh_2_rect(n_inv_pos, n_inv_sz)
                                    rectnpred = np.array(rectnpred)
                                    b_overlap_nold_pred = [poly_iou(rectnpred, n_pred) for n_pred in _location_neighbor_pred]
                                    print(b_overlap_nold_inv)
                                    print(b_overlap_nold_pred)
                                    b_n_2pass = (np.array(b_overlap_nold_inv)+np.array(b_overlap_nold_pred))/2
                                    b_n_2pass = [x for x in b_n_2pass]
                                    b_overlap_nold_inv = b_n_2pass
                                    print(b_overlap_nold_inv)
                                    '''
                                costmatrix[ind+1,]=b_overlap_nold_inv

                            cost_matrix = self.scorematrix2costmatrix(costmatrix)
                            #from munkres import Munkres, print_matrix

                            Mun = Munkres()
                            Munindexes = Mun.compute(cost_matrix)
                            #print_matrix(matrix, msg='Lowest cost through this matrix:')
                            #print(Munindexes)
                            
                            minusmatrix = copy.deepcopy(costmatrix)*-1
                            #print(costmatrix)
                            #print(minusmatrix)
                            row, column = Munindexes[0]
                            
                            
                            #print('*2  max(costmatrix)- costmatrix = cost_matrix = ',cost_matrix)
                            #print('Munindexes = ',Munindexes)
                            try:
                                value = costmatrix[row][column]
                                #print('value = ',value)
                                minusmatrix[row][column] = value
                            except:
                                pass
                            #print('minusmatrix = ',minusmatrix)
                            all_inv_iou = minusmatrix[0,:]


                            '''
                            argmax_y=999
                            while(np.max(costmatrix)!=-1):
                                #print(costmatrix)
                                [argmax_y,argmax_x] =np.unravel_index(costmatrix.argmax(), costmatrix.shape)

                                #print(argmax_x)
                                #print(argmax_y)
                                if argmax_y == 0:
                                    break
                                costmatrix[argmax_y,:]=-1
                                costmatrix[:,argmax_x]=-1
                            all_inv_iou = costmatrix[0,:]
                            '''







                        #print(costmatrix)
                        all_inv_iou = np.array(all_inv_iou)
                        
                        if not gt is None:
                            postp1=[]
                            sztp1=[]
                            postp1.append(state['target_pos'])
                            sztp1.append(state['target_sz'])
                            for _ind_nms_pred in nms_bbox_index:
                                postp1.append([state['n_target_pos'][0][_ind_nms_pred],state['n_target_pos'][1][_ind_nms_pred]])
                                sztp1.append([state['n_target_sz'][0][_ind_nms_pred],state['n_target_sz'][1][_ind_nms_pred]])
                            if kalman_neighbor_mode:
                                postp1.append(state['KF_predict_pos'])
                                sztp1.append(state['old_target_sz'][-1])
                            b_overlap_tp1=[]
                            for _ind_tp1_pred in range(len(postp1)):
                                location_postp1 = cxy_wh_2_rect(postp1[_ind_tp1_pred], sztp1[_ind_tp1_pred])
                                b_overlap_tp1.append(poly_iou(gt, location_postp1))
                            b_overlap_tp1 = np.array(b_overlap_tp1)
                            print('b_overlap_tp1 = ',b_overlap_tp1)
                            print('all_inv_iou = ',all_inv_iou)
                            
                            if all_inv_iou.max()>inv_error_th:
                                if all_inv_iou.argmax() == b_overlap_tp1.argmax():
                                    choose_right_pred = choose_right_pred+1
                                    state['flag_max_iou_after_method']=1

                            else:
                                #if kalman_neighbor_mode and all_inv_iou.argmax() == b_overlap_tp1.argmax() and all_inv_iou.argmax()==len(all_inv_iou)-1: wrong bf 220805
                                if kalman_neighbor_mode and all_inv_iou.argmax() == b_overlap_tp1.argmax() and all_inv_iou.argmax()==(KF_index-1+1):

                                    choose_right_pred = choose_right_pred+1
                                    state['flag_max_iou_after_method']=1

                                else:
                                    location_nowkf = cxy_wh_2_rect(state['KF_predict_pos'], state['old_target_sz'][-1])
                                    iou_gt_nowkf = poly_iou(gt, location_nowkf)
                                    if iou_gt_nowkf>max(b_overlap_tp1):
                                        choose_right_pred = choose_right_pred+1
                                        state['flag_max_iou_after_method']=1
                            print('frame {}, now right pick/all pick = {}/{} = {}%'.format(now_frame_index,choose_right_pred,warning_2pass,(choose_right_pred/warning_2pass)*100.0))
                            print('frame {}, now model pick/all pick = {}/{} = {}%'.format(now_frame_index,max_iou_in_model_pred,warning_2pass,(max_iou_in_model_pred/warning_2pass)*100.0))
                            #if state['flag_max_iou_in_model_pred']==1 and state['flag_max_iou_after_method']==0:
                            if state['flag_max_iou_in_model_pred']==0 and state['flag_max_iou_after_method']==1:
                                state['flag_output_image']=1
                            else:
                                state['flag_output_image']=1
                            
                        if all_inv_iou.max()>inv_error_th:
                            print('all_inv_iou.argmax() = ',all_inv_iou.argmax())
                            print('all_inv_iou = ',all_inv_iou)
                            
                            if all_inv_iou.argmax()==0:
                                (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                                state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                                state['choose']='model pred1'
                                if group_inv:
                                    location_neighbor_pos=location_neighbor_pos[1:]
                                    location_neighbor_sz=location_neighbor_sz[1:]
                                    location_neighbor_posind=location_neighbor_posind[1:]

                                
                            else:
                                
                                if ls_add_mode:
                                    if not group_inv:
                                        inv_pos = state['inv_target_pos'].copy()
                                        inv_sz = state['inv_target_sz'].copy()
                                        inv_posind = state['inv_target_posind'].copy()

                                        inv_pos.append(state['target_pos'])
                                        inv_sz.append(state['target_sz'])
                                        inv_posind.append(now_frame_index)
                                        
                                        if len(location_neighbor_pos)==0:
                                            location_neighbor_pos = np.array([inv_pos])
                                            location_neighbor_sz = np.array([inv_sz])
                                            location_neighbor_posind = np.array([inv_posind])
                                        else:
                                            #print('l_posv3 = ',[inv_pos])
                                            location_neighbor_pos = np.append(location_neighbor_pos,[inv_pos],0)
                                            location_neighbor_sz = np.append(location_neighbor_sz,[inv_sz],0)
                                            location_neighbor_posind = np.append(location_neighbor_posind,[inv_posind],0)
                                else:
                                    if group_inv:
                                        location_neighbor_pos=location_neighbor_pos[1:]
                                        location_neighbor_sz=location_neighbor_sz[1:]
                                        location_neighbor_posind=location_neighbor_posind[1:]

                                #if kalman_neighbor_mode and all_inv_iou.argmax()==len(all_inv_iou)-1 :
                                if kalman_neighbor_mode and all_inv_iou.argmax()==(KF_index-1+1) :

                                    state['target_pos'] = state['KF_predict_pos']
                                    state['target_sz'] = state['old_target_sz'][-1]
                                    #state['score'] = np.float64(1)# use original score
                                    
                                    #b_overlap_KFpred = poly_iou(gt[f], location_KF_pred)
                                    #b_overlap = b_overlap_KFpred
                                    state['KF_est_pos'] = state['target_pos']
                                    state['choose']='KF1'

                                else:
                                    #print(kalman_neighbor_mode)
                                    #print(all_inv_iou.argmax()-1)
                                    #print(KF_index-1)
                                    #print(len(nms_bbox_index))
                                    now_max_iou_ind = nms_bbox_index[all_inv_iou.argmax()-1]

                                    state['target_pos'][0] = state['n_target_pos'][0][now_max_iou_ind]
                                    state['target_pos'][1] = state['n_target_pos'][1][now_max_iou_ind]

                                    state['target_sz'][0] = state['n_target_sz'][0][now_max_iou_ind]
                                    state['target_sz'][1] = state['n_target_sz'][1][now_max_iou_ind]

                                    state['score'] = state['n_score'][now_max_iou_ind]

                                    (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
                                    state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
                                    state['choose']='neighbor0'
                                #b_overlap = all_overlap.max()
                        else:
                            #print(location_neighbor_pos)
                            #print(state['target_pos'])
                            if kalman_add_mode or ls_add_mode:
                                if not group_inv:
                                    #print('not implement')
                                    inv_pos = state['inv_target_pos'].copy()
                                    inv_sz = state['inv_target_sz'].copy()
                                    inv_posind = state['inv_target_posind'].copy()

                                    inv_pos.append(state['target_pos'])
                                    inv_sz.append(state['target_sz'])
                                    inv_posind.append(now_frame_index)






                                    if len(location_neighbor_pos)==0:
                                        location_neighbor_pos = np.array([inv_pos])
                                        location_neighbor_sz = np.array([inv_sz])
                                        location_neighbor_posind = np.array([inv_posind])
                                    else:
                                        #print('l_posv2 = ',[inv_pos])

                                        location_neighbor_pos = np.append(location_neighbor_pos,[inv_pos],0)
                                        location_neighbor_sz = np.append(location_neighbor_sz,[inv_sz],0)
                                        location_neighbor_posind = np.append(location_neighbor_posind,[inv_posind],0)
                            else:
                                if group_inv:
                                    location_neighbor_pos=location_neighbor_pos[1:]
                                    location_neighbor_sz=location_neighbor_sz[1:]
                                    location_neighbor_posind=location_neighbor_posind[1:]
 

                            #print(location_neighbor_pos)



                            state['target_pos'] = state['KF_predict_pos']
                            state['target_sz'] = state['old_target_sz'][-1]
                            #use original score
                            #state['score'] = np.float64(1)
                            #b_overlap_KFpred = poly_iou(gt[f], location_KF_pred)
                            #b_overlap = b_overlap_KFpred
                            state['KF_est_pos'] = state['target_pos']
                            state['choose']='KF2'





        else:
            if state['inviou'] < _warning_inv_th:
                state['nowconsecutive'] = state['nowconsecutive']+1
            else:
                state['nowconsecutive'] = 0
            if not history_dict is None:
                if history_dict['nowconsecutive']>0:
                    history_dict['nowconsecutive'] = copy.copy(state['nowconsecutive'])
                
            state['flag_output_image']=0
            state['choose']='model pred2'
            (kf_est_x,kf_est_y) = state['KF'].update(np.array([state['target_pos'][0],state['target_pos'][1]]))
            state['KF_est_pos'] = np.squeeze(np.asarray([kf_est_x,kf_est_y]))[0]
            '''
            #before211104 if main tracker > th then not inv neighbors

            if len(location_neighbor_pos)>0:
                #print(123123123)
                #print('location_neighbor_pos = ',location_neighbor_pos)
                _location_neighbor_pos = []
                _location_neighbor_sz = []
                for ind in range(len(location_neighbor_pos)):
                    _location_neighbor_pos.append([np.array(location_neighbor_pos[ind])])
                    _location_neighbor_sz.append([np.array(location_neighbor_sz[ind])])
                location_neighbor_pos = _location_neighbor_pos
                location_neighbor_sz = _location_neighbor_sz
            '''
            
            if len(location_neighbor_pos)>0:
                if not group_inv:
                    l_pos_,l_sz_,l_posind_,l_score_=\
                     self.inv_neighbor_single(im,state['oldim'],location_neighbor_pos,location_neighbor_sz,\
                      state,now_frame_index,inv_start_from_history,old_target_pos_sz)
                else:
                    l_pos_,l_sz_,l_posind_,l_score_=\
                     self.inv_neighbor_group(im,state['oldim'],location_neighbor_pos,\
                                             location_neighbor_sz,location_neighbor_score,\
                                             state,now_frame_index,inv_start_from_history,old_target_pos_sz,\
                                             neighbor_score_only,neighbor_static_th,neighbor_th)

                for ind in range(len(location_neighbor_pos)):                        
                    '''
                    #inverse frames from model output not miximize target pos
                    if 'online_tracker_inv' in state:
                        n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,state['oldim'],np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],online_tracker_inv=state['online_tracker_inv'],dataname=state['dataname'],resume=state['resume'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
                    else:
                        n_inv_pos, n_inv_sz, n_inv_score,n_inv_index = self.track_inv(im,state['oldim'],np.array(location_neighbor_pos[ind]),np.array(location_neighbor_sz[ind]),state['net'],state['p'],now_frame_index = now_frame_index,inv_start_from_history=inv_start_from_history,old_target_pos_sz=old_target_pos_sz)
                    #print('location_neighbor_pos[ind]',location_neighbor_pos[ind])
                    #print('n_inv_pos = ',n_inv_pos)
                    '''
                    #l_pos = l_pos_[ind].copy()
                    #l_sz = n_inv_sz.copy()
                    #l_posind=n_inv_index.copy()
                    l_pos_[ind].append(np.array(location_neighbor_pos[ind]))
                    l_sz_[ind].append(np.array(location_neighbor_sz[ind]))
                    l_posind_[ind].append(np.array(location_neighbor_posind[ind]))
                    
                    #print('n_inv_index = ',n_inv_index)
                    #print('np.array(location_neighbor_posind[ind]) = ',np.array(location_neighbor_posind[ind]))
                    #print('now frame  = ',now_frame_index)
                    #print('len(oldim) = ',len(state['oldim']))
                    #print('l_posv1 = ',l_pos)
                    #print('inv_frames = ',inv_frames)
                    location_neighbor_pos[ind] = l_pos_[ind]
                    location_neighbor_sz[ind] = l_sz_[ind]
                    location_neighbor_posind[ind] = l_posind_[ind]
        
        
        
        if dynamic_ig==2:
            if 'KF' in state['choose']:
                _flag=0
            else:
                _flag=1
                
            (state['use_flag_next'],state['need_del_index_next']) = self.update_need_del_index(state['use_flag'],_flag,inv_frames)

            
            
        if kalman_neighbor_mode and KF_index>0:
            #print('len(location_neighbor_pos) = ',len(location_neighbor_pos))
            #print('len(location_neighbor_posind) = ',len(location_neighbor_posind))
            #location_neighbor_pos[KF_index-1] = []
            #location_neighbor_sz[KF_index-1] = []
            #location_neighbor_posind[KF_index-1] = []
            
            location_neighbor_pos = location_neighbor_pos[:KF_index-1]+location_neighbor_pos[KF_index-1+1:]
            location_neighbor_sz = location_neighbor_sz[:KF_index-1]+location_neighbor_sz[KF_index-1+1:]
            location_neighbor_posind = location_neighbor_posind[:KF_index-1]+location_neighbor_posind[KF_index-1+1:]
            
            # wrong bf 220805
            #location_neighbor_pos = location_neighbor_pos[:-1]
            #location_neighbor_sz = location_neighbor_sz[:-1]
            #location_neighbor_posind = location_neighbor_posind[:-1]

            
            
        state['old2_neighbor_pos']=copy.deepcopy(state['old_neighbor_pos'])
        state['old2_neighbor_sz']=copy.deepcopy(state['old_neighbor_sz'])
        state['old2_neighbor_posind']=copy.deepcopy(state['old_neighbor_posind'])

            
        state['old_neighbor_pos']=copy.deepcopy(location_neighbor_pos)
        state['old_neighbor_sz']=copy.deepcopy(location_neighbor_sz)
        state['old_neighbor_posind']=copy.deepcopy(location_neighbor_posind)



        state['warning_2pass'] = warning_2pass
        state['choose_right_pred'] = choose_right_pred
        state['max_iou_in_model_pred'] = max_iou_in_model_pred
        state['max_iou_in_kalman_pred'] = max_iou_in_kalman_pred
        state['max_iou_in_neighbor_pred'] = max_iou_in_neighbor_pred
        state['location_neighbor_pred'] = location_neighbor_pred
        state['location_pred'] = location_pred
        state['location_KF_pred'] = location_KF_pred
        state['all_inv_iou'] = all_inv_iou
        if not gt is None:
            state['gt']=gt
        
        return state
    
    
    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)



class Ocean(object):
    def __init__(self, info):
        super(Ocean, self).__init__()
        self.info = info   # model and benchmark info
        self.stride = 8
        self.align = info.align
        self.online = info.online
        self.trt = info.TRT

    def init(self, im, target_pos, target_sz, model, hp=None):
        # in: whether input infrared image
        state = dict()
        # epoch test
        p = OceanConfig()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # single test
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT', 'GOT10K', 'LASOT'] if x in self.info.dataset]
            if len(prefix) == 0: prefix = [self.info.dataset]
            absPath = os.path.abspath(os.path.dirname(__file__))
            yname = 'Ocean.yaml'
            yamlPath = os.path.join(absPath, '../../experiments/test/{0}/'.format(prefix[0]), yname)
            cfg = load_yaml(yamlPath)
            if self.online:
                temp = self.info.dataset + 'ON'
                cfg_benchmark = cfg[temp]
            else:
                cfg_benchmark = cfg[self.info.dataset]
            p.update(cfg_benchmark)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()

        # double check
        # print('======= hyper-parameters: penalty_k: {}, wi: {}, lr: {}, ratio: {}, instance_sz: {}, score_sz: {} ======='.format(p.penalty_k, p.window_influence, p.lr, p.ratio, p.instance_size, p.score_size))

        # param tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = hp['big_sz']
                p.renew()
            else:
                p.instance_size = hp['small_sz']
                p.renew()

        if self.trt:
            print('====> TRT version testing: only support 255 input, the hyper-param is random <====')
            p.instance_size = 255
            p.renew()

        self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = z_crop.unsqueeze(0)
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones((int(p.score_size), int(p.score_size)))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p):

        if self.align:
            cls_score, bbox_pred, cls_align = net.track(x_crops)

            cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
            cls_align = F.sigmoid(cls_align).squeeze().cpu().data.numpy()
            cls_score = p.ratio * cls_score + (1- p.ratio) * cls_align

        else:
            cls_score, bbox_pred = net.track(x_crops)
            cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2-pred_x1, pred_y2-pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2-pred_x1) / (pred_y2-pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        if self.online_score is not None:
            s_size = pscore.shape[0]
            o_score = cv2.resize(self.online_score, (s_size, s_size), interpolation=cv2.INTER_CUBIC)
            pscore = p.online_ratio * o_score + (1 - p.online_ratio) * pscore
        else:
            pass

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, state, im, online_score=None, gt=None):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        if online_score is not None:
            self.online_score = online_score.squeeze().cpu().data.numpy()
        else:
            self.online_score = None

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        x_crop = x_crop.unsqueeze(0)

        target_pos, target_sz, target_score = self.update(net, x_crop.cuda(), target_pos, target_sz*scale_z, window, scale_z, p)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p
        state['score'] = target_score
        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class OceanConfig(object):
    penalty_k = 0.062
    window_influence = 0.38
    lr = 0.765
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
    context_amount = 0.5
    ratio = 0.94


    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
