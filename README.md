# NeighborTrack
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighbortrack-improving-single-object/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=neighbortrack-improving-single-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighbortrack-improving-single-object/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=neighbortrack-improving-single-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighbortrack-improving-single-object/visual-object-tracking-on-trackingnet)](https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=neighbortrack-improving-single-object)

The implement of [**NeighborTrack: Improving Single Object Tracking by Bipartite Matching with Neighbor Tracklets**](https://arxiv.org/abs/2211.06663)

Single Object Tracking post processing method by using cycle consistency and neighbor(python version)  

some code are from OSTrack, Votchallenge, Ocean, TransT, pytracking, Mixformer. thanks these projects alot.

[**OSTrack:**](https://github.com/botaoye/OSTrack)

[**TransT:**](https://github.com/chenxin-dlut/TransT)

[**Votchallenge:**](https://github.com/votchallenge/toolkit)

[**Ocean:**](https://github.com/JudasDie/SOTS/tree/master)

[**pytracking:**](https://github.com/visionml/pytracking)

[**Mixformer**](https://github.com/MCG-NJU/MixFormer)

[**KalmanFilter implement:**](https://github.com/RahmadSadli/2-D-Kalman-Filter)

[**SoftNMS implement:**](https://github.com/bharatsingh430/soft-nms)

## [**Demo videos**](https://www.youtube.com/playlist?list=PLhJHN1Q0397Kr1n-3Zs084Wn0KPPL_s47)
## [**Models and source results link**](https://drive.google.com/drive/folders/1GXyEdmwkyfPP7oKoSAcFfYTuXzWwG5ch?usp=share_link)
[**More information for model paths**](https://github.com/franktpmvu/NeighborTrack/blob/main/model_download_setting.md)
## Python Environment 
```shell
pip install munkres==1.1.4
pip install shapely

```
Other dependencies depend on your base model, e.g. [**OSTrack:**](https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/example_ostrack_install.sh)


# Get result from NeighborTrack with OSTrack
Work space are in NeighborTrack/trackers/ostrack/ , please remember change dataset and model's [**root**]( https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/lib/test/evaluation/local.py)
Please seen [**OSTrack user's guide**](https://github.com/botaoye/OSTrack) to set project paths

## LaSOT,GOT10K,TrackingNet (baseline from OSTrack github code)
|LaSOT|AUC|OP50|OP75|Precision|Norm Precision|
|---|---|---|---|---|---|
|OSTrack384| 71.90      | 82.91      | 72.50      | 77.65        | 81.40             |
|OSTrack384_NeighborTrack| 72.25      | 83.33      | 72.70      | 78.05        | 81.82             |
			
|[**GOT-10K**](http://got-10k.aitestunion.com/leaderboard)|AO|SR0.50|SR0.75|Hz|
|---|---|---|---|---|
|OSTrack384| 73.94|	83.63|	72.16|	7.00 fps|
|OSTrack384_NeighborTrack| 75.73|	85.72|	73.29|	2.99 fps|
|OSTrack384_gottrainonly| 74.19|	83.98|	71.58|  3.88 fps|
|OSTrack384_gottrainonly_NeighborTrack| 74.53|	84.25|	71.54|	4.07 fps|


|[**TrackingNet**](https://eval.ai/web/challenges/challenge-page/1805/leaderboard/4225)|Success|Precision|Normalized Precision|Coverage| 
|---|---|---|---|---|
|OSTrack384| 83.58      | 82.94      | 88.05      | 100        |
|OSTrack384_NeighborTrack| 83.79      | 83.24      | 88.30      | 100        |



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
to get vot challenge result, please seen detail by code [**trackers.ini**](https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/trackers.ini),  [**ostrack_384_vot_neighbor.py**](https://github.com/franktpmvu/NeighborTrack/blob/main/trackers/ostrack/tracking/ostrack_384_vot_neighbor.py)

[**setting example**](https://github.com/franktpmvu/NeighborTrack/blob/main/example_vot_toolkit.sh)

if you want to know how to create workspace of vot2022st vot2020 vot2021 dataset, please seen [**Votchallenge:**](https://github.com/votchallenge/toolkit)

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

## [**More details:**](https://github.com/franktpmvu/NeighborTrack/blob/main/CreateNeededFunction.md)


