# [**Download link(google drive)**](https://drive.google.com/drive/folders/1GXyEdmwkyfPP7oKoSAcFfYTuXzWwG5ch?usp=share_link)


AR model are from :

https://github.com/MCG-NJU/MixFormer/tree/main/external/AR, 

https://github.com/MasterBin-IIAU/AlphaRefine

put "ARnet_seg_mask_ep0040.pth.tar" on path  NeighborTrack/trackers/ostrack/pytracking/networks

OSTrack model are from :
https://github.com/botaoye/OSTrack

# LaSOT

put "OSTrack_ep0300.pth.tar" on both path  

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_ep300_neighbor/

and

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_ep300/

# GOT-10K(OSTrack only training on got10k train - set)

put "OSTrack_ep0100.pth.tar" on both path  

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_got10k_ep100_neighbor/

and

NeighborTrack/trackers/ostrack/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_got10k_ep100/

# Update dataset path
please update NeighborTrack/trackers/ostrack/lib/test/evaluation/local.py
[**Guideline**](https://github.com/botaoye/OSTrack)
