# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import _init_paths
import os
import cv2
import torch
import random
import argparse
import numpy as np
import sys
from os.path import join, isdir, isfile
from os import makedirs
from os import rename
import math
import json

try:
    from torch2trt import TRTModule
except:
    print('Warning: TensorRT is not successfully imported')

import models.models as models

from os.path import exists, join, dirname, realpath
from tracker.ocean import Ocean,Ocean2pass
from tracker.online import ONLINE
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
from utils.siammaskDetectUtils import softnms_cpu_torch


from eval_toolkit.pysot.datasets import VOTDataset
from eval_toolkit.pysot.evaluation import EAOBenchmark

#print(os.getcwd())
#print(sys.path)

from core.eval_otb import eval_auc_tune
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description='Test Ocean')
    parser.add_argument('--arch', dest='arch', default='OceanTRT', help='backbone architecture')
    parser.add_argument('--resume', default="snapshot/OceanV.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--align', default='True', type=str, help='alignment module flag') # bool 
    parser.add_argument('--online', action='store_true',help='online flag')
    parser.add_argument('--onlineinv', action='store_true', help='inverse online flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--optvideo',action='store_true', help='output_video')
    parser.add_argument('--optvideo-path', dest='optvideo_path',default='./', type=str, help='save video path')
    parser.add_argument('--optimg',action='store_true', help='output_image_frame')
    parser.add_argument('--ISFH',action='store_true', help='inv_start_from_history')
    parser.add_argument('--optimg-path', dest='optimg_path',default='./', type=str, help='save video path')
    parser.add_argument('--model-nickname', default='', help='nickname,will add after epoch')
    parser.add_argument('--th-warning-inv', default=0.5, help='nickname,will add after epoch')
    parser.add_argument('--th-nms', default=0.25, help='nickname,will add after epoch')
    parser.add_argument('--nms-sigma', default=0.01, help='nms-sigma if ++ then get more bboxes ')
    parser.add_argument('--th-KF', default=0.0, help='less than =>KF')
    parser.add_argument('--neighbor', default=0, help='use t-1 neighbor and t neighbor matching , neighbor iou=IOU(neighbor,inv(targetbox))')
    parser.add_argument('--neighbor-predandinv', default=0, help='neighbor iou = IOU(pred,targetbox) + IOU(neighbor,inv(targetbox))')
    parser.add_argument('--trust-one-target', default=1, help='if just one target in predict, choose it')
    parser.add_argument('--KF-add-neighbor', default=0, help='if KF ,add pred to neighbor')
    parser.add_argument('--ls-add-neighbor', default=0, help='if KF ,add pred to neighbor')
    parser.add_argument('--max-consecutive', default=-1, help='if more then -1 , will use model predict')
    parser.add_argument('--delay-start',action='store_true', help='if history bar of bbox not full then use pred ans')
    parser.add_argument('--group-inv',action='store_true', help='grouping inverse to hungarian')
    parser.add_argument('--inv-frames', default=1, help='inverse check what frames')
    parser.add_argument('--ignore-frames', default=0, help='ignore frames from t=now to t=old')
    parser.add_argument('--dynamic-ignore', default=0, help='dynamic ignore jump less then th-warning-inv s frame')
    parser.add_argument('--kalman-neighbor-mode', default=0, help='add kalman in neighbor to inverse by apperiance feature')
    parser.add_argument('--time-dilate', default=0, help='add dilate in temporal domain')
    parser.add_argument('--pred-margin-ratio', default=1.0, help='max pred multiple ratio e.g.*1.1')
    parser.add_argument('--neighbor-search-ratio', default=1.0, help='search space of neighbor ratio e.g.*2.0')
    parser.add_argument('--neighbor-score-only', action='store_true', help='window= 1 * w or 0 default = 1 * w')
    parser.add_argument('--neighbor-static-th', action='store_true', help='use static th 0.7 (True) or max(x)*0.7(false)')
    parser.add_argument('--neighbor-th', default=0.7, help='use neighbor >th ')
    parser.add_argument('--histdict', action='store_true', help='save some history as npy')
    args = parser.parse_args()


    return args


def argmax1dto2d(_1d,h,w):
    #argmax_1d = np.argmax(costmatrix)
    #costmatrix=np.zeros((len(state['old_neighbor_pos'])+1,len(location_neighbor_pos)+1))
    y = math.floor((_1d+1)/w)
    x = (_1d+1)-w*y-1
    #print(argmax_x)
    #print(argmax_y)

    return x,y
    

def reloadTRT():
    absPath = os.path.abspath(os.path.dirname(__file__))
    t_bk_path = join(absPath, '../', 'snapshot', 't_backbone.pth')
    s_bk_siam255_path = join(absPath, '../', 'snapshot', 's_backbone_siam255.pth')
    s_bk_siam287_path = join(absPath, '../', 'snapshot', 's_backbone_siam287.pth')
    s_bk_online_path = join(absPath, '../', 'snapshot', 's_backbone_online.pth')
    t_neck_path = join(absPath, '../', 'snapshot', 't_neck.pth')
    s_neck255_path = join(absPath, '../', 'snapshot', 's_neck255.pth')
    s_neck287_path = join(absPath, '../', 'snapshot', 's_neck287.pth')
    multiDiCorr255_path = join(absPath, '../', 'snapshot', 'multiDiCorr255.pth')
    multiDiCorr287_path = join(absPath, '../', 'snapshot', 'multiDiCorr287.pth')
    boxtower255_path = join(absPath, '../', 'snapshot', 'boxtower255.pth')
    boxtower287_path = join(absPath, '../', 'snapshot', 'boxtower287.pth')

    t_bk = TRTModule()
    s_bk_siam255 = TRTModule()
    s_bk_siam287 = TRTModule()
    s_bk_online = TRTModule()
    t_neck = TRTModule()
    s_neck255 = TRTModule()
    s_neck287 = TRTModule()
    multiDiCorr255 = TRTModule()
    multiDiCorr287 = TRTModule()
    boxtower255 = TRTModule()
    boxtower287 = TRTModule()

    t_bk.load_state_dict(torch.load(t_bk_path))
    s_bk_siam255.load_state_dict(torch.load(s_bk_siam255_path))
    s_bk_siam287.load_state_dict(torch.load(s_bk_siam287_path))
    s_bk_online.load_state_dict(torch.load(s_bk_online_path))
    t_neck.load_state_dict(torch.load(t_neck_path))
    s_neck255.load_state_dict(torch.load(s_neck255_path))
    s_neck287.load_state_dict(torch.load(s_neck287_path))
    multiDiCorr255.load_state_dict(torch.load(multiDiCorr255_path))
    multiDiCorr287.load_state_dict(torch.load(multiDiCorr287_path))
    boxtower255.load_state_dict(torch.load(boxtower255_path))
    boxtower287.load_state_dict(torch.load(boxtower287_path))

    return [t_bk, s_bk_siam255, s_bk_siam287, s_bk_online, t_neck, s_neck255, s_neck287, multiDiCorr255, multiDiCorr287, boxtower255, boxtower287]


def init_history_dict(video_name,history_dict=None):
    if history_dict is None:
        history_dict=dict()
    history_dict['nowname'] = video_name
    history_dict['nowconsecutive'] = 0
    history_dict[video_name]=dict()
    history_dict[video_name]['frame']=[]
    history_dict[video_name]['nowinviou']=[]
    history_dict[video_name]['GTIOU']=[]
    history_dict[video_name]['pos_sz']=[]
    history_dict[video_name]['score']=[]
    history_dict[video_name]['invweight']=[]
    history_dict[video_name]['invweightGT']=[]
    history_dict[video_name]['invweightHistGT']=[]
    history_dict[video_name]['consecutive']=[]

    
    return history_dict


def get_inv_ious():
    return []
    
def track_ori(siam_tracker, online_tracker, siam_net, video, args):
    start_frame, toc = 0, 0

    # save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', args.dataset, args.arch + suffix)
    else:
        tracker_path = os.path.join('result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    lost = 0

    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            state = siam_tracker.init(im, target_pos, target_sz, siam_net)  # init tracker

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            if args.online:
                state = online_tracker.track(im, rgb_im, siam_tracker, state)
            else:
                state = siam_tracker.track(state, im)

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
                lost += 1
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        elif 'OTB' in args.dataset or 'LASOT' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VISDRONE' in args.dataset or 'GOT10K' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))

def track_exist(video):
    if args.epoch_test or args.model_nickname:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', args.dataset, args.arch + suffix + args.model_nickname)
    else:
        tracker_path = os.path.join('result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting


def track(siam_tracker, online_tracker, siam_net, video,args,online_tracker_inv=None,history_dict=None):
    #print(args)
    if not history_dict is None:
        hist_d=history_dict

    start_frame, toc = 0, 0

    # save result to evaluate
    if args.epoch_test or args.model_nickname:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', args.dataset, args.arch + suffix + args.model_nickname)
    else:
        tracker_path = os.path.join('result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    lost = 0
    warning_2pass = 0
    #warning_old_less_then_pred = 0
    max_iou_in_model_pred = 0
    max_iou_in_kalman_pred = 0
    max_iou_in_neighbor_pred = 0
    
    
    if args.th_warning_inv=='dy':
        warning_inv_th = args.th_warning_inv
    else:
        warning_inv_th = np.float64(args.th_warning_inv)
    inv_error_th = np.float64(args.th_KF) #less than =>KF
    gtchoose=0 # use gt to choose final answer
    just_kalman=0 #always use kalman to choose final answer
    neighbor_2pass_mode=np.int64(args.neighbor) #use t-1 neighbor and t neighbor matching
    neighbor_predandinv_mode=np.int64(args.neighbor_predandinv) # neighbor iou = IOU(pred,targetbox) + IOU(neighbor,inv(targetbox))
    one_target_pred_first=np.int64(args.trust_one_target) # if just one target in predict, choose it
    kalman_add_mode = np.int64(args.KF_add_neighbor) #if KF ,add pred to neighbor
    ls_add_mode = np.int64(args.ls_add_neighbor)
    inv_frames = np.int64(args.inv_frames)
    nms_th = np.float64(args.th_nms)
    neighbor_th=np.float64(args.neighbor_th) #use t-1 neighbor and t neighbor matching

    if nms_th==0.0:
        nms_th=nms_th-0.01
    nms_sigma = np.float64(args.nms_sigma)
    ignore_frames = np.int64(args.ignore_frames)
    max_consecutive = np.int64(args.max_consecutive)
    pred_margin_ratio = np.float64(args.pred_margin_ratio)
    neighbor_search_ratio = np.float64(args.neighbor_search_ratio)

    print('pred_margin_ratio = ',pred_margin_ratio)
    
    if args.dynamic_ignore =='warninginv':
        dynamic_ig = 1
    if args.dynamic_ignore =='KF':
        dynamic_ig = 2
        
    if args.dynamic_ignore =='0':
        dynamic_ig = 0
    if args.dynamic_ignore =='1':
        dynamic_ig = 1
    
    dynamic_ignore = np.int64(dynamic_ig)
    
    
    kalman_neighbor_mode = np.int64(args.kalman_neighbor_mode)
    time_dilate = np.int64(args.time_dilate)

    image_files, gt = video['image_files'], video['gt']
    
    if args.optvideo:
        im = cv2.imread(image_files[0])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if not isdir(args.optvideo_path): makedirs(args.optvideo_path)

        outV = cv2.VideoWriter(args.optvideo_path+args.dataset+'_'+video['name']+'.avi', fourcc, 10.0, (im.shape[1], im.shape[0]))
        #print(outV.isOpened())
    if args.optimg:
        #if not isdir(args.optimg_path): makedirs(args.optimg_path)
        if not isdir(args.optimg_path+args.dataset+'_'+video['name']+'/'): makedirs(args.optimg_path+args.dataset+'_'+video['name']+'/')

        fp = open(args.optimg_path+args.dataset+'_'+video['name']+'.txt', "w")


    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            '''
            #e35 tune
            hp = dict()#eao0.322
            hp['penalty_k'] = 0.041
            hp['window_influence'] = 0.34800000000000003
            hp['lr'] = 0.674
            hp['small_sz'] = 255
            hp['big_sz'] = 319
            hp['ratio'] = 0.92
            
            state = siam_tracker.init(im, target_pos, target_sz, siam_net,hp)  # init tracker
            '''
            if args.online:
                state = siam_tracker.init(im, target_pos, target_sz, siam_net,online_tracker_inv = online_tracker_inv,dataname= args.dataset,resume=args.resume,warning_inv_th=warning_inv_th,gt=gt[f])  # init tracker
                #print('online_tracker_inv = ',online_tracker_inv)
                #print('state[online_tracker_inv] = ',state['online_tracker_inv'])
            else:
                state = siam_tracker.init(im, target_pos, target_sz, siam_net,warning_inv_th=warning_inv_th,gt=gt[f])  # init tracker
            #print('----------------------- init state is -------------------------')
            #print(state)
            #print('----------------------- end -------------------------')

            if args.online:
                online_tracker.init(im, rgb_im, siam_net, target_pos, target_sz, True, dataname=args.dataset, resume=args.resume)

            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            if args.online:
                neighbortrack_args=dict()
                neighbortrack_args['warning_inv_th'] = warning_inv_th
                neighbortrack_args['inv_error_th'] = inv_error_th
                neighbortrack_args['gtchoose'] = gtchoose
                neighbortrack_args['just_kalman'] = just_kalman
                neighbortrack_args['neighbor_2pass_mode'] = neighbor_2pass_mode
                neighbortrack_args['neighbor_predandinv_mode'] = neighbor_predandinv_mode
                neighbortrack_args['one_target_pred_first'] = one_target_pred_first
                neighbortrack_args['kalman_add_mode'] = kalman_add_mode
                neighbortrack_args['ls_add_mode'] = ls_add_mode
                neighbortrack_args['inv_frames'] = inv_frames
                neighbortrack_args['ignore_frames'] = ignore_frames
                neighbortrack_args['kalman_neighbor_mode'] = kalman_neighbor_mode
                neighbortrack_args['time_dilate'] = time_dilate
                neighbortrack_args['gt'] = gt[f]
                neighbortrack_args['pass_check_mode'] = False
                neighbortrack_args['dynamic_ig'] = dynamic_ignore
                neighbortrack_args['now_frame_index'] = f
                neighbortrack_args['nms_th'] = nms_th
                neighbortrack_args['nms_sigma'] = nms_sigma
                neighbortrack_args['delay_start'] = args.delay_start
                neighbortrack_args['inv_start_from_history'] = args.ISFH
                neighbortrack_args['max_consecutive'] = max_consecutive
                neighbortrack_args['pred_margin_ratio'] = pred_margin_ratio
                neighbortrack_args['neighbor_search_ratio'] = neighbor_search_ratio
                neighbortrack_args['neighbor_score_only'] = args.neighbor_score_only
                neighbortrack_args['neighbor_static_th'] = args.neighbor_static_th
                neighbortrack_args['neighbor_th'] = neighbor_th
                neighbortrack_args['group_inv'] = args.group_inv
                
                if not history_dict is None:
                    neighbortrack_args['history_dict'] = hist_d
                
                state = online_tracker.neighbor_track(im, rgb_im, siam_tracker, state,neighbortrack_args)
            else:
                state = siam_tracker._neighbor_track(state, im, warning_inv_th=warning_inv_th,inv_error_th=inv_error_th,gtchoose=gtchoose,just_kalman=just_kalman,neighbor_2pass_mode=neighbor_2pass_mode,neighbor_predandinv_mode=neighbor_predandinv_mode,one_target_pred_first=one_target_pred_first,kalman_add_mode=kalman_add_mode,ls_add_mode=ls_add_mode,inv_frames=inv_frames,ignore_frames=ignore_frames,kalman_neighbor_mode=kalman_neighbor_mode,time_dilate=time_dilate, gt=gt[f],pass_check_mode=False,dynamic_ig=dynamic_ignore,now_frame_index=f,nms_th=nms_th,nms_sigma=nms_sigma,delay_start=args.delay_start,inv_start_from_history=args.ISFH,history_dict=hist_d,max_consecutive=max_consecutive,pred_margin_ratio=pred_margin_ratio,neighbor_search_ratio=neighbor_search_ratio,neighbor_score_only=args.neighbor_score_only,neighbor_static_th=args.neighbor_static_th,neighbor_th=neighbor_th,group_inv=args.group_inv)

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

            
            
            
            if 'VOT' in args.dataset:
                b_overlap = poly_iou(gt[f], location)


            else:
                b_overlap = 1
            
            #location_KF_est = cxy_wh_2_rect(state['KF_est_pos'], state['pred_target_sz'])

            #if state['inviou'] < warning_inv_th:
            #    warning_2pass += 1
            #    b_overlap_old = poly_iou(gt[f], location_old)
            #    if b_overlap>b_overlap_old:
            #        warning_old_less_then_pred += 1
                    
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
                lost += 1

        else:
            regions.append(0)
            
            

        toc += cv2.getTickCount() - tic
        
        if args.optvideo or args.optimg:
            im_show = im.copy()
            im_show_hist = im.copy()
            im_show_neighbor_inv = im.copy()
            im_show_old_neighbor_inv = im.copy()
            im_show_pred_inv = im.copy()
            im_show_choose = im.copy()
            im_show_choose_inv = im.copy()
            im_show_original = im.copy()
            im_show_KF = im.copy()
            
            
            if f == 0: 
                cv2.destroyAllWindows()
            if f > start_frame or f == start_frame-5:
                warning_2pass = state['warning_2pass']
                #warning_old_less_then_pred = 0
                max_iou_in_model_pred = state['max_iou_in_model_pred']
                max_iou_in_kalman_pred = state['max_iou_in_kalman_pred']
                max_iou_in_neighbor_pred = state['max_iou_in_neighbor_pred']
                
                location_final = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                location_neighbor_pred = state['location_neighbor_pred']
                location_n_pos = state['old_neighbor_pos'].copy()
                location_n_sz = state['old_neighbor_sz'].copy()
                location_o_pos = state['old_target_pos'].copy()
                location_o_sz = state['old_target_sz'].copy()
                location_inv_pos = state['inv_target_pos'].copy()
                location_inv_sz = state['inv_target_sz'].copy()
                
                location_oldn_pos = state['old2_neighbor_pos'].copy()
                location_oldn_sz = state['old2_neighbor_sz'].copy()


                location_pred = state['location_pred']# model predict
                location_KF_pred = state['location_KF_pred']
                all_inv_iou = state['all_inv_iou']
                
                
                if len(location_pred) == 8:
                    if mask_enable:
                        mask = mask > state['p'].seg_thr
                        im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                        
                        
                    for locations_ind in range(len(location_n_pos)):
                        npos = location_n_pos[locations_ind]
                        nsz = location_n_sz[locations_ind]
                        for lind2 in range(len(npos)):
                            location_n_rect = cxy_wh_2_rect(npos[lind2],nsz[lind2] )
                            location_n_rect = np.int0(location_n_rect)
                            cv2.polylines(im_show, [location_n_rect.reshape((-1, 1, 2))], True, (0, 0, 128), 3)

                        
                    for location_n_pred in location_neighbor_pred:
                        location_int_n = np.int0(location_n_pred)
                        cv2.polylines(im_show, [location_int_n.reshape((-1, 1, 2))], True, (0, 0, 255), 3)

                        
                    for locations_ind in range(len(location_o_pos)):
                        location_o_rect = cxy_wh_2_rect(location_o_pos[locations_ind],location_o_sz[locations_ind] )
                        location_o_rect = np.int0(location_o_rect)
                        cv2.polylines(im_show, [location_o_rect.reshape((-1, 1, 2))], True, (128, 0, 0), 3)

                        
                        
                    location_int = np.int0(location_pred)
                    cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
                    
                    
                    #location_old_int = np.int0(location_old)
                    #cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 0, 0), 3)
                    
                    #location_inv_int = np.int0(location_inv)
                    #cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (255, 255, 255), 3)

                    
                    location_inv_int = np.int0(location_KF_pred)
                    cv2.polylines(im_show_KF, [location_int.reshape((-1, 1, 2))], True, (255, 0, 255), 3)
                    
                    #location_inv_int = np.int0(location_KF_est)
                    #cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (128, 255, 128), 3)

                    
                else:

                    ind_n=0
                    if 'use_flag' in state:
                        use_flag = state['use_flag']
                    else:
                        use_flag = None
                        
                        
                        
                    for locations_ind in range(len(location_n_pos)):
                        npos = location_n_pos[locations_ind]
                        nsz = location_n_sz[locations_ind]
                        for lind2 in range(len(npos[:-1])):
                            location_n_rect = cxy_wh_2_rect(npos[lind2],nsz[lind2])
                            location_n_rect = np.int0(location_n_rect)
                            if not use_flag is None:
                                if use_flag[lind2]:
                                    cv2.rectangle(im_show_neighbor_inv, (location_n_rect[0], location_n_rect[1]),
                                          (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 0, 128), 3)
                            else:
                                cv2.rectangle(im_show_neighbor_inv, (location_n_rect[0], location_n_rect[1]),
                                          (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 0, 128), 3)

                                
                    for locations_ind in range(len(location_oldn_pos)):
                        npos = location_oldn_pos[locations_ind]
                        nsz = location_oldn_sz[locations_ind]
                        for lind2 in range(len(npos[:-1])):
                            location_n_rect = cxy_wh_2_rect(npos[lind2],nsz[lind2])
                            location_n_rect = np.int0(location_n_rect)
                            if not use_flag is None:
                                if use_flag[lind2]:
                                    cv2.rectangle(im_show_old_neighbor_inv, (location_n_rect[0], location_n_rect[1]),
                                          (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 255, 255), 3)
                            else:
                                cv2.rectangle(im_show_old_neighbor_inv, (location_n_rect[0], location_n_rect[1]),
                                          (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 255, 255), 3)



                    for location_n_pred in location_neighbor_pred:
                        ind_n += 1 
                        location_int_n = [int(l) for l in location_n_pred]
                        cv2.rectangle(im_show, (location_int_n[0], location_int_n[1]),
                                  (location_int_n[0] + location_int_n[2], location_int_n[1] + location_int_n[3]), (0, 0, 255), 3)
                        cv2.rectangle(im_show_original, (location_int_n[0], location_int_n[1]),
                                  (location_int_n[0] + location_int_n[2], location_int_n[1] + location_int_n[3]), (0, 0, 255), 3)
                        all_inv_iou = np.array(all_inv_iou)

                        if len(all_inv_iou)>0:
                            cv2.putText(im_show, str('%.3f'%all_inv_iou[ind_n]), (location_int_n[0]+10, location_int_n[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            cv2.putText(im_show_original, str('%.3f'%all_inv_iou[ind_n]), (location_int_n[0]+10, location_int_n[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
                            
                            
                        
                    for locations_ind in range(len(location_inv_pos)):
                        location_inv_rect = cxy_wh_2_rect(location_inv_pos[locations_ind],location_inv_sz[locations_ind] )
                        location_inv_rect = np.int0(location_inv_rect)
                        if not use_flag is None:
                            if use_flag[locations_ind]:
                                cv2.rectangle(im_show_pred_inv, (location_inv_rect[0], location_inv_rect[1]),
                                          (location_inv_rect[0] + location_inv_rect[2], location_inv_rect[1] + location_inv_rect[3]), (0, 0, 128), 3)
                        else:
                            cv2.rectangle(im_show_pred_inv, (location_inv_rect[0], location_inv_rect[1]),
                                      (location_inv_rect[0] + location_inv_rect[2], location_inv_rect[1] + location_inv_rect[3]), (0, 0, 128), 3)


                    
                    for locations_ind in range(len(location_o_pos)):
                        location_o_rect = cxy_wh_2_rect(location_o_pos[locations_ind],location_o_sz[locations_ind] )
                        location_o_rect = np.int0(location_o_rect)
                        if not use_flag is None:
                            if use_flag[locations_ind]:
                                cv2.rectangle(im_show_hist, (location_o_rect[0], location_o_rect[1]),
                                      (location_o_rect[0] + location_o_rect[2], location_o_rect[1] + location_o_rect[3]), (128, 0, 0), 3)
                        else:
                            cv2.rectangle(im_show_hist, (location_o_rect[0], location_o_rect[1]),
                                  (location_o_rect[0] + location_o_rect[2], location_o_rect[1] + location_o_rect[3]), (128, 0, 0), 3)


                    
                    
                    
                    
                    
                    location = [int(l) for l in location_KF_pred]
                    cv2.rectangle(im_show_KF, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (255, 0, 255), 3)

                    location = [int(l) for l in location_pred]
                    cv2.rectangle(im_show, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (0, 0, 255), 3)
                    
                    location = [int(l) for l in location_pred]
                    cv2.rectangle(im_show_original, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (255, 0, 0), 3)

                    #cv2.putText(im_show, str('%.3f'%state['inviou']), (location[0]+10, location[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    location = [int(l) for l in location_final]
                    cv2.rectangle(im_show_choose, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (0, 0, 255), 3)

                    location = [int(l) for l in location_final]
                    cv2.rectangle(im_show, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (255, 0, 0), 3)

                    
                    for locations_ind in range(len(location_n_pos)):
                        npos = location_n_pos[locations_ind]
                        nsz = location_n_sz[locations_ind]
                        npos0=npos[-1]
                        nsz0=nsz[-1]
                        location_n_rect0 = cxy_wh_2_rect(npos0,nsz0)
                        b_overlap_choose_inv = poly_iou(np.array(location_final), np.array(location_n_rect0))
                        if b_overlap_choose_inv==1:
                            print(npos)
                            for lind2 in range(len(npos[:-1])):
                                print(npos[lind2])
                                location_n_rect = cxy_wh_2_rect(npos[lind2],nsz[lind2])
                                location_n_rect = np.int0(location_n_rect)
                                if not use_flag is None:
                                    if use_flag[lind2]:
                                        cv2.rectangle(im_show_choose_inv, (location_n_rect[0], location_n_rect[1]),
                                                  (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 0, 128), 3)
                                else:
                                    cv2.rectangle(im_show_choose_inv, (location_n_rect[0], location_n_rect[1]),
                                          (location_n_rect[0] + location_n_rect[2], location_n_rect[1] + location_n_rect[3]), (0, 0, 128), 3)


                    

                    

                    #ind_n=0
                    
                    #for location_n_pred in location_neighbor_pred:
                    #    ind_n += 1 
                    #    if len(all_inv_iou)>0:
                    #        cv2.putText(im_show, str('%.3f'%all_inv_iou[ind_n]), (location_int_n[0]+10, location_int_n[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


                    
                    #location = [int(l) for l in location_old]
                    #cv2.rectangle(im_show, (location[0], location[1]),
                    #              (location[0] + location[2], location[1] + location[3]), (0, 0, 0), 3)
                    
                    #location = [int(l) for l in location_inv]
                    #cv2.rectangle(im_show, (location[0], location[1]),
                    #              (location[0] + location[2], location[1] + location[3]), (255, 255, 255), 3)


                    #location = [int(l) for l in location_KF_est]
                    #cv2.rectangle(im_show, (location[0], location[1]),
                    #              (location[0] + location[2], location[1] + location[3]), (128, 0, 128), 3)

            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    cv2.polylines(im_show_original, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    try:
                        cv2.rectangle(im_show, (int(gt[f, 0]), int(gt[f, 1])), (int(gt[f, 0]) + int(gt[f, 2]), int(gt[f, 1]) + int(gt[f, 3])), (0, 255, 0), 3)
                        cv2.rectangle(im_show_original, (int(gt[f, 0]), int(gt[f, 1])), (int(gt[f, 0]) + int(gt[f, 2]), int(gt[f, 1]) + int(gt[f, 3])), (0, 255, 0), 3)
                    except:
                        print(gt[f,])
                        break

            cv2.putText(im_show, 'frame: '+str(f), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(im_show_original, 'frame: '+str(f), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, str(lost), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'forward score: ' + str(state['score'])[:4] if 'score' in state else '', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            #cv2.putText(im_show, 'overlap with GT: '+str(b_overlap)[:4] if 'score' in state else '', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'invIOU: '+str(state['inviou']) if 'score' in state else '', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            #cv2.putText(im_show, 'invIOU<'+str(warning_inv_th)+' :'+str(warning_2pass), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            #cv2.putText(im_show, +str(warning_old_less_then_pred), (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'old_target_score: '+str(state['old_target_score'])[:4] if 'old_target_score' in state else '', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'inv_target_score: '+str(state['inv_target_score'])[:4] if 'inv_target_score' in state else '', (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            
            
            

            #cv2.putText(im_show, 'model_win = '+str(max_iou_in_model_pred), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'kalman_win = '+str(max_iou_in_kalman_pred), (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            #cv2.putText(im_show, 'neighbor_win = '+str(max_iou_in_neighbor_pred), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            '''
            choose_right_pred = state['choose_right_pred']
            max_iou_in_model_pred = state['max_iou_in_model_pred']
            if warning_2pass>0:
                cv2.putText(im_show, 'choose_right_pred/warning: '+str((choose_right_pred/warning_2pass)*100.0), (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(im_show, 'max_iou_in_model_pred/warning: '+str((max_iou_in_model_pred/warning_2pass)*100.0), (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            cv2.putText(im_show, 'choose: '+state['choose'], (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            '''

            #cv2.imshow(video['name'], im_show)
            if args.optvideo:
                outV.write(im_show)
                #print(123)
            if args.optimg and state['flag_output_image']:
            #if args.optimg:
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'.jpg',im_show)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_KF.jpg',im_show_KF)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_hist.jpg',im_show_hist)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_ninv.jpg',im_show_neighbor_inv)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_oldninv.jpg',im_show_old_neighbor_inv)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_predinv.jpg',im_show_pred_inv)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_choose.jpg',im_show_choose)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_choose_inv.jpg',im_show_choose_inv)
                cv2.imwrite(args.optimg_path+args.dataset+'_'+video['name']+'/'+'%06d'%(f)+'_ori.jpg',im_show_original)
                fp.write('{}, {}, {}, {}, {}\n'.format(str(f),str(lost),str(state['score'])[:4] if 'score' in state else 0,str(b_overlap)[1:4] if 'score' in state else 0,str(state['inviou_ori'])[1:4] if 'inviou_ori' in state else 0))
            cv2.waitKey(1)
    if args.optvideo:
        outV.release()
        os.rename(args.optvideo_path+args.dataset+'_'+video['name']+'.avi', args.optvideo_path+args.dataset+'_'+video['name']+'_loss_'+str(lost)+'.avi')
    #cv2.destroyAllWindows()



    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        elif 'OTB' in args.dataset or 'LASOT' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
        elif 'VISDRONE' in args.dataset or 'GOT10K' in args.dataset:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    

    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {} max_iou_model {} max_iou_kalman {} max_iou_neighbor {}'.format(video['name'], toc, f / toc, lost,max_iou_in_model_pred,max_iou_in_kalman_pred,max_iou_in_neighbor_pred))
    if args.optimg:
        fp.write('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {} max_iou_model {} max_iou_kalman {} max_iou_neighbor {}'.format(video['name'], toc, f / toc, lost,max_iou_in_model_pred,max_iou_in_kalman_pred,max_iou_in_neighbor_pred))
        fp.close()
    if not history_dict is None:
        return hist_d


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.TRT = 'TRT' in args.arch
    info.epoch_test = args.epoch_test

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.online = args.online
    siam_info.epoch_test = args.epoch_test
    siam_info.TRT = 'TRT' in args.arch
    if args.online:
        siam_info.align = False
    else:
        siam_info.align = True if 'VOT' in args.dataset and args.align=='True' else False

    if siam_info.TRT:
        siam_info.align = False

    #siam_tracker = Ocean(siam_info)
    siam_tracker = Ocean2pass(siam_info)
    
    siam_net = models.__dict__[args.arch](align=siam_info.align, online=args.online)
    print(siam_net)
    print('===> init Siamese <====')

    if not siam_info.TRT:
        siam_net = load_pretrain(siam_net, args.resume)
    else:
        print("tensorrt toy model: not loading checkpoint")
    siam_net.eval()
    siam_net = siam_net.cuda()

    if siam_info.TRT:
        print('===> load model from TRT <===')
        print('===> please ignore the warning information of TRT <===')
        print('===> We only provide a toy demo for TensorRT. There are some operations are not supported well.<===')
        print('===> If you wang to test on benchmark, please us Pytorch version. <===')
        print('===> The tensorrt code will be contingously optimized (with the updating of official TensorRT.)<===')
        trtNet = reloadTRT()
        siam_net.tensorrt_init(trtNet)

    if args.online:
        online_tracker = ONLINE(info)
        if args.onlineinv:
            online_tracker_inv = ONLINE(info)
            print('args.onlineinv = ',args.onlineinv)
            print('online_tracker_inv init done')
        else:
            print('args.onlineinv = ',args.onlineinv)
            online_tracker_inv = None
    else:
        print('now is offline mode')
        online_tracker = None
        online_tracker_inv = None

    print('====> warm up <====')
    for i in tqdm(range(100)):
        siam_net.template(torch.rand(1, 3, 127, 127).cuda())
        siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()
    
    if args.video is not None:
        track(siam_tracker, online_tracker, siam_net, dataset[args.video], args,online_tracker_inv=online_tracker_inv)
    else:
        ii=0
        init_hist=0
        for video in video_keys:
            if args.histdict and init_hist==0:
                print(dataset[video]['name'])
                hist_d = init_history_dict(dataset[video]['name'])
                init_hist=1
            else:
                hist_d = init_history_dict(dataset[video]['name'],history_dict=hist_d)

            if args.histdict:
                hist_d = track(siam_tracker, online_tracker, siam_net, dataset[video], args,online_tracker_inv=online_tracker_inv,history_dict = hist_d)
                if not isdir(args.optimg_path+args.dataset+'_'+dataset[video]['name']+'/'):
                    makedirs(args.optimg_path+args.dataset+'_'+dataset[video]['name']+'/')
                #print(hist_d)
                hist_now = open(args.optimg_path+args.dataset+'.json','w')
                json.dump(hist_d,hist_now,cls=NumpyEncoder)
                hist_now.close()

            else:
                track(siam_tracker, online_tracker, siam_net, dataset[video], args,online_tracker_inv=online_tracker_inv)


            ii=ii+1
            print('video :{} =  {}/{}'.format(video,ii,len(video_keys)))
            
        if args.histdict:
            if not os.path.isfile(args.optimg_path+args.dataset+'_fin.json'):
                hist_now = open(args.optimg_path+args.dataset+'_fin.json','w')
                json.dump(hist_d,hist_now,cls=NumpyEncoder)
                hist_now.close()


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# -----------------------------------------------
# The next few functions are utilized for tuning
# -----------------------------------------------
def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                     '_small_size_{:.4f}'.format(hp['small_sz']) +
                                     '_big_size_{:.4f}'.format(hp['big_sz']) +
                                     '_ratio_{:.4f}'.format(hp['ratio']) +
                                     '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                     '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                     '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB'):
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K'):
            return 0
        else:
            print('benchmark not supported now')
            return

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]

    # for rgbt splited test

    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net, hp=hp)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VISDRONE' in benchmark_name  or 'GOT10K' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'VIS' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name:
        return tracker_path
    else:
        print('benchmark not supported now')


def auc_otb(tracker, net, config):
    """
    get AUC for OTB benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc

def eao_vot(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    re_path = result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    # debug
    print('======> debug: results_path')
    print(result_path)
    print(os.system("ls"))
    print(join(realpath(dirname(__file__)), '../dataset'))

    # give abs path to json path
    data_path = join(realpath(dirname(__file__)), '../dataset')
    dataset = VOTDataset(config['benchmark'], data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    return eao


if __name__ == '__main__':
    main()

