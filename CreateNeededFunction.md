# How to create the 3 functions needed:

## def initialize(self,image,init_info):
initialize the tracking method. init state xywh img or something else.
Depends on what kind of initialization your SOT model needs. Please don't init cuda model in this block,
cuda model shold init outside of `def initialize:`, e.g. `__init__()`,
initialize means change target or init target in video, not init model.
`return []`
please see ostrack example :https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/lib/test/tracker/ostrack.py#L50

## def track_neighbor(self,image,th):
this function need to get the candidates from your own SOT tracker,
you can get candidates from score > max(C)*0.7 or whatever.
Watch the different of "track_neighbor" and "track".
1."track"have some code need update tracker's state,
Do Not Update That in "track_neighbor", e.g. update position, center, template, ...,
For example, ncc tracker need update the tracking center of image,
please do not update it in `def track_neighbor:`,
because update tracking center of image will changed the answer of next call `track_neighbor`.
If any code will change answer of call `track_neighbor` when input same image and xywh,
please put it in `def update_center:` 
2. Output candidates and it's score by SOT tracker, we didn't need features of candidates.
Not only one way can get candidates, for example, the top 10 score BBOX are also fine.
Dont forgot: Much candidates Much Slower. 
 

`return xywh, score, neighborxywh, neighborscore`
### xywh 
trackers 'original' answer xywh == `[x,y,w,h]`
### score 
trackers 'original' answer score == `score`
### neighbor xywh
candidate xywh in cell == `[[(C1)x,y,w,h],[(C2)x,y,w,h],...]`
### neighbor score
candidate score in cell == `[(C1)score,(C2)score,...]`

https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/lib/test/tracker/ostrack.py#L149

## def update_center(self,xywh):
update teplate, DIMP, center, learningrate, train model,..., etc.
Please put it on this block.
Our method will update it after get that frame's final answer (xywh).
`return []`
https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/ostrack/lib/test/tracker/ostrack.py#L192
# See more example: https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/example_TransT.py#L12
https://github.com/franktpmvu/NeighborTrack/blob/c889695427a2288b42e31cd0f9e0f7e509244729/trackers/example_ncc_tracker.py#L51
