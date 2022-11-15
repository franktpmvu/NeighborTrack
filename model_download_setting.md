# [**Download link(google drive)**](https://drive.google.com/drive/folders/1GXyEdmwkyfPP7oKoSAcFfYTuXzWwG5ch?usp=share_link)


AR model are from :

https://github.com/MCG-NJU/MixFormer/tree/main/external/AR, 

https://github.com/MasterBin-IIAU/AlphaRefine

put "ARnet_seg_mask_ep0040.pth.tar" on path  NeighborTrack/trackers/ostrack/pytracking/networks

OSTrack model are from :
https://github.com/botaoye/OSTrack

---------lasot----------

put "OSTrack_ep0300.pth.tar" on both path  

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_ep300_neighbor/

and

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_ep300/

---------got10k_only_train_got10k----------

put "OSTrack_ep0100.pth.tar" on both path  

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_got10k_ep100_neighbor/

and

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_got10k_ep100/


please update NeighborTrack/trackers/ostrack/lib/test/evaluation/local.py
guideline: https://github.com/botaoye/OSTrack
