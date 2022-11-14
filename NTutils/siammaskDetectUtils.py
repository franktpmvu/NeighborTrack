import glob
import sys
import numpy as np
import torch
import time 
import cv2


def area_of(left_top, right_bottom):

    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
        return types: torch.Tensor
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def softnms_cpu_torch(box_scores, score_threshold=0.001, sigma=0.5, top_k=-1,output_index=0):
    """Soft NMS implementation.
    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    box_index_=torch.tensor(np.arange(1,box_scores.size(0)+1,1)-1)
    #print(box_scores.size(0))
    if output_index==1:
        pickedIndex=[]
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = box_scores[max_score_index, :].clone()
        picked_box_scores.append(cur_box_prob.numpy())
        #print(box_index_[max_score_index])
        #print(box_index_)

        if output_index ==1:
            pickedIndex.append(box_index_[max_score_index.item()].item())
            #print(max_score_index)
            #print(box_index_)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            #print('--------------------')
            #print(picked_box_scores)
            #print(len(picked_box_scores) == top_k > 0)
            #print(box_scores.size(0))
            #print(box_scores.size(0) == 1)
            #print('--------------------')
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        box_index_[max_score_index] = box_index_[-1]
        box_index_ = box_index_[:-1]
        
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        #print([box_scores[:, -1] > score_threshold])
        #print(max_score_index)
        #print(box_index_)

        box_index_ = box_index_[box_scores[:, -1] > score_threshold]
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    #print(score_threshold)
    #print(box_scores.size(0))
    if len(picked_box_scores) > 0:
        if output_index==1:
            return picked_box_scores,pickedIndex
        else:
            return torch.stack(picked_box_scores)
    
    else:
        if output_index==1:
            #print('here0')
            return torch.tensor([]),torch.tensor([])
        else:
            #print('here1')
            return torch.tensor([])


def input_add_del_bottom():
    while(True):
        flag = cv2.waitKey(1)
        if flag==27:
            input_ans=0
            return input_ans
        if flag==ord('i'):
            input_ans=1
            return input_ans

        if flag==ord('d'):
            input_ans=-1
            return input_ans


def input_bbox(ims):
    print('please choose bboxes input(i) or delete(d)')
    input_ans=input_add_del_bottom()
    x=[]
    y=[]
    w=[]
    h=[]
    flagInputBbox=100
    #print(ims[0])
    while(flagInputBbox!=0):
        try:
            init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
            x_now, y_now, w_now, h_now = init_rect
            flagInputBbox=np.sum(init_rect,axis=0)
            if flagInputBbox!=0:
                x.append(x_now)
                y.append(y_now)
                w.append(w_now)
                h.append(h_now)

            #print(init_rect)
            #print(flagInputBbox)
        except:
            print('you are in here')
            exit()
    
    return x,y,w,h,input_ans

def xyxyMultiRatio(bboxWithScore,ratio):
    alldata=[]
    for data in bboxWithScore:
        data[0]=data[0]*ratio
        data[1]=data[1]*ratio
        data[2]=data[2]*ratio
        data[3]=data[3]*ratio
        alldata.append(data)
    return alldata

def bbox2DetFormat(bboxWithScore,txtfile,nowframe):
    for data in bboxWithScore:
        #xyxy to xywh
        data[2]=data[2]-data[0]
        data[3]=data[3]-data[1]

        data = ['%03f'%x for x in data]
        
        #print(data)
        #temp = '%d'%data
        #print(['%d'%nowframe+',-1,'+data[0]+','+data[1]+','+data[2]+','+data[3]+','+data[4]+'\n'])
        txtfile.write('%d'%nowframe+',-1,'+data[0]+','+data[1]+','+data[2]+','+data[3]+','+data[4]+'\n')
