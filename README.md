# NeighborTrack
Single Object Tracking post processing method by using cycle consistency and neighbor  
KalmanFilter are from https://github.com/RahmadSadli/2-D-Kalman-Filter

some code are from OSTrack,Votchallenge,Ocean,TransT thanks alot.

OSTrack:https://github.com/botaoye/OSTrack
TransT:https://github.com/chenxin-dlut/TransT
Votchallenge:https://github.com/votchallenge/toolkit
Ocean:https://github.com/JudasDie/SOTS/tree/master

How to use in your own SOT tracker:
please see the neighbortrack.py there are a simple code from Votchallenge NCC tracker , add 3 function to use our method.
    def initialize(self,image,init_info):
      return []
    def track_neighbor(self,image,th):
      return {trackers 'original' answer xywh == <[x,y,w,h]>},{trackers 'original' answer score == <score>},{CAND_xywh in cell == [[(C1)x,y,w,h],[(C2)x,y,w,h],...]}, {cand score in cell ==[(c1)score,(c2)score,...]}
    def update_center(self,xywh):


