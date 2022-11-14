# NeighborTrack
The implement of NeighborTrack: Improving Single Object Tracking by Bipartite Matching with Neighbor Tracklets

Single Object Tracking post processing method by using cycle consistency and neighbor(python version)  

some code are from OSTrack,Votchallenge,Ocean,TransT,pytracking thanks these projects alot.

OSTrack:https://github.com/botaoye/OSTrack
TransT:https://github.com/chenxin-dlut/TransT
Votchallenge:https://github.com/votchallenge/toolkit
Ocean:https://github.com/JudasDie/SOTS/tree/master
pytracking:https://github.com/visionml/pytracking
KalmanFilter are from https://github.com/RahmadSadli/2-D-Kalman-Filter
SoftNMS are from https://github.com/bharatsingh430/soft-nms

## Demo videos
https://www.youtube.com/playlist?list=PLhJHN1Q0397Kr1n-3Zs084Wn0KPPL_s47
## Models and source results
https://drive.google.com/drive/folders/1GXyEdmwkyfPP7oKoSAcFfYTuXzWwG5ch?usp=share_link

## Python Dependent 
```shell
pip install munkres==1.1.4
```
Other dependencies depend on your base model, e.g. OSTrack:
https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/example_ostrack_install.sh


# Get result from NeighborTrack with OSTrack
Work space are in NeighborTrack/trackers/ostrack/ , please remember change dataset and model's root in https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/lib/test/evaluation/local.py. please seen user's guided in OSTrack:https://github.com/botaoye/OSTrack Set project paths

## LaSOT,GOT10K
|LaSOT|AUC|OP50|OP75|Precision|Norm Precision|
|---|---|---|---|---|---|
|OSTrack384| 71.90      | 82.91      | 72.50      | 77.65        | 81.40             |
|OSTrack384_NeighborTrack| 72.25      | 83.33      | 72.70      | 78.05        | 81.82             |
			
|GOT-10K|AO|SR0.50|SR0.75|Hz|
|---|---|---|---|---|
|OSTrack384| 0.739|	0.836|	0.722|	7.00 fps|
|OSTrack384_NeighborTrack| 0.757|	0.857|	0.733|	2.99 fps|
|OSTrack384_gottrainonly| 0.741|	0.839|	0.715|  3.88 fps|
|OSTrack384_gottrainonly_NeighborTrack| 0.745|	0.842|	0.715|	4.07 fps|



```shell 
cd /your_path/trackers/ostrack/
sh test.sh
#or
#lasot example
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset lasot --threads 24 --num_gpus 8 --neighbor 1
#python tracking/analysis_results.py 

#got-10K example
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_test --threads 16 --num_gpus 8 --neighbor 1 
#to use got-10K train_from_got10K_only
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100_neighbor --dataset got10k_test --threads 16 --num_gpus 8 --neighbor 1 

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

```shell
vot test ostrackNeighbor
vot test ostrackNeighborAR
vot evaluate --workspace ./vot2022st ostrackNeighbor
vot analysis --workspace vot2022st ostrackNeighbor

vot evaluate --workspace ./vot2021 ostrackNeighborAR
vot analysis --workspace vot2021 ostrackNeighborAR
```
please seen detail from NeighborTrack/trackers.ini, NeighborTrack/trackers/ostrack/tracking/ostrack_384_vot_neighbor.py 

if you want to know how to create workspace of vot2022st vot2020 vot2021 dataset, please seen Votchallenge:https://github.com/votchallenge/toolkit

## In your own video
```shell 
sh video_test.sh
# or
python tracking/video_demo_neighbor.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor ./cup1.avi  \
   --optional_box 1109 531 82 135 --save_results --debug 1 --save_img
#optional_box is GT in first frame.
```

# How to use NeighborTrack in your own SOT tracker:

## init and use NeighborTracker:
### init
https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/tracking/ostrack_384_vot_neighbor.py#L63
### get tracking answer
https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/tracking/ostrack_384_vot_neighbor.py#L70


tracker and invtracker is original ostrack, you can change it by your SOT tracker.

region = `[x,y,w,h]`,(x y = top left) 

image = image by your model input, for example ostrack's image = `numpy.array(img[h,w,3(RGB])`

there are a simple code from Votchallenge NCC tracker , add 3 function to use our method.(`initialize`, `track_neighbor` and `update_center`).
Please see: https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/example_ncc_tracker.py#L14  
After add functions are seems like: https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/example_ncc_tracker.py#L51

Remenber ,the tracker should be have 2 indepandent model forward/reverse, because all of SOT method will forgot tracking target after initialize , if just 1 forward/backward tracker, it cannot switch forward/backward mission and ansure forward answer don't have any change (even didn't use our method to change output, just use same tracker to track any other object, your forward output will not comeback to original answer, because memory of tracker are changed.) 

## other example: ostrack add 3 functions https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/lib/test/evaluation/tracker.py#L328

## more details:
https://github.com/franktpmvu/NeighborTrack/blob/main/CreateNeededFunction.md


