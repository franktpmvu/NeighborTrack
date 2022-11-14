#python tracking/video_demo.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor /data/OSTrack/crabs1.avi  \
#   --optional_box 20 596 97 59 --save_results --debug 1 --save_img
#python tracking/video_demo.py ostrack vitb_384_mae_ce_32x4_ep300 /data/OSTrack/chameleon6.avi  \
#   --optional_box 25 407 183 99 --save_results --debug 1 --save_img
#python tracking/video_demo.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor /data/OSTrack/cup1.avi  \
#   --optional_box 1109 531 82 135 --save_results --debug 1 --save_img
python tracking/video_demo_neighbor.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor ./cup1.avi  \
   --optional_box 1109 531 82 135 --save_results --debug 1 --save_img
