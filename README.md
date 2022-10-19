# NeighborTrack
Single Object Tracking post processing method by using cycle consistency and neighbor  

some code are from OSTrack,Votchallenge,Ocean,TransT thanks these projects alot.

OSTrack:https://github.com/botaoye/OSTrack
TransT:https://github.com/chenxin-dlut/TransT
Votchallenge:https://github.com/votchallenge/toolkit
Ocean:https://github.com/JudasDie/SOTS/tree/master

KalmanFilter are from https://github.com/RahmadSadli/2-D-Kalman-Filter

## Dependent 
```shell
pip install munkres==1.1.4
```
other dependent on your base model e.g. OSTrack

# Get result from NeighborTrack with OSTrack
in space of NeighborTrack/trackers/ostrack/ , please remember change dataset and model's root in `NeighborTrack/trackers/ostrack/lib/test/evaluation/local.py`. please seen user's guided in OSTrack:https://github.com/botaoye/OSTrack Set project paths

## LaSOT,GOT10K
|LaSOT|AUC|OP50|OP75|Precision|Norm Precision|
|---|---|---|---|---|---|
|OSTrack384| 71.90      | 82.91      | 72.50      | 77.65        | 81.40             |
|OSTrack384_NeighborTrack| 72.25      | 83.33      | 72.70      | 78.05        | 81.82             |
			
|GOT-10K|AO|SR0.50|SR0.75|Hz|
|---|---|---|---|---|
|OSTrack384| 0.739|	0.836|	0.722|	7.00 fps|
|OSTrack384_NeighborTrack| 0.757|	0.857|	0.733|	2.99 fps|


fps are not sure, server cpu&gpu always full ...
```shell 
sh test.sh
#or
#lasot example
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset lasot --threads 24 --num_gpus 8 --neighbor 1
#got-10K example
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_test --threads 24 --num_gpus 8 --neighbor 1 
```
## votchallenge
|VOT2022-ST|EAO|A|R|
|---|---|---|---|
|OSTrack384| 0.538|	0.779|	0.824|
|OSTrack384_NeighborTrack| 0.564|	0.779|	0.845|
|Ocean| 0.484|	0.703|	0.823|
|Ocean_NeighborTrack| 0.486|	0.703|	0.822|
|TransT_N2| 0.493|	0.780|	0.775|
|TransT_N2_NeighborTrack| 0.519|	0.781|	0.808|
|TransT_N4| 0.486|	0.779|	0.771|
|TransT_N4_NeighborTrack| 0.518|	0.777|	0.810|
|Normal Cross Correlation tracker(NCC)| 0.102|	0.564|	0.208|
|NCC_NeighborTrack| 0.127|	0.549|	0.266|

others vot2020 vot2021 please seen paper

```shell
vot test ostrackNeighborgit
vot evaluate --workspace ./vot2022st ostrackNeighborgit
vot analysis --workspace vot2022st ostrackNeighborgit
```
please seen detail from NeighborTrack/trackers.ini, NeighborTrack/trackers/ostrack/tracking/ostrack_384_vot_neighbor.py 

if you want to know how to create workspace of vot2022st vot2020 vot2021 dataset, please seen Votchallenge:https://github.com/votchallenge/toolkit

## in your own video
```shell 
sh video_test.sh
# or
python tracking/video_demo_neighbor.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor ./cup1.avi  \
   --optional_box 1109 531 82 135 --save_results --debug 1 --save_img
#optional_box is GT in first frame.
```

# How to use NeighborTrack in your own SOT tracker:
please see the neighbortrack.py there are a simple code from Votchallenge NCC tracker , add 3 function to use our method.(`initialize`, `track_neighbor` and `update_center`). Remenber ,the tracker should be have 2 indepandent model forward/inverse, because all of SOT method will forgot tracking target after initialize , if just 1 forward/backward tracker, it cannot switch forward/backward mission and ansure forward answer don't have any change (even didn't use our method to change output, just use same tracker to track any other object, your forward output will not comeback to original answer, because memory of tracker are changed.) 

to seen ostrack with our method, please seen NeighborTrack/trackers/ostrack/lib/test/evaluation/tracker.py ```class NeighborTracker(Tracker):```



## def initialize(self,image,init_info):
initialize the tracking method. init state xywh img or something else. please don't init cuda model in this block, cuda model 
  shold init outside of initialize, e.g. `__init__()`, initialize means change target or init target in video, not init model.
`return []`
## def track_neighbor(self,image,th):
this function need to get the CAND from your own SOT tracker, you can get CANDs from score >max(C)*0.7 or whatever, watch different of "track_neighbor" and "track", do not update position,center,template,...,etc. in track_neighbor if any code will change answer of call track_neighbor when input same image and xywh, please put it on update_center. 

`return xywh, score, neighborxywh, neighborscore`
#### xywh 
trackers 'original' answer xywh == `[x,y,w,h]`
#### score 
trackers 'original' answer score == `score`
#### neighbor xywh
CAND_xywh in cell == `[[(C1)x,y,w,h],[(C2)x,y,w,h],...]`
#### neighbor score
CAND score in cell == `[(c1)score,(c2)score,...]`
## def update_center(self,xywh):
update teplate, DIMP, center, learningrate, train model,..., etc. please put it on this block. 
`return []`

## init and use NeighborTracker:

```python
ntracker = neighbortrack(tracker,image,region[:2],region[2:],invtracker=invtracker)
for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
    image = self._read_image(frame_path)
    start_time = time.time()
    info = seq.frame_info(frame_num)
    info['previous_output'] = prev_output
    if len(seq.ground_truth_rect) > 1:
        info['gt_bbox'] = seq.ground_truth_rect[frame_num]
                
                
    #out = tracker.track(image, info)
        
    state = ntracker._neighbor_track(image)
    location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
    out = {"target_bbox": location}
```

tracker and invtracker is original ostrack, you can change it by yours.
region = `[x,y,w,h]`
image = image by your model input, for ostrack, it is `numpy.array(img[h,w,3(RGB])`

