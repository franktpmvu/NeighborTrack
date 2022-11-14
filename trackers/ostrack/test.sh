
#----lasot----

#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset lasot --threads 24 --num_gpus 8 --neighbor 1
#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 0 --num_gpus 1
#python tracking/analysis_results.py 
#----got10k----


#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset got10k_test --threads 8 --num_gpus 4
#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_test --threads 15 --num_gpus 5 --neighbor 1 
#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset got10k_val --threads 15 --num_gpus 5 

#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_val --threads 24 --num_gpus 8 --neighbor 1 
#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_test --threads 24 --num_gpus 8 --neighbor 1 

#python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100_neighbor --dataset got10k_test --threads 24 --num_gpus 8 --neighbor 1


#python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name ostrack384_got10k_elimination_ep100_neighbor_n9

#----trackingnet----

#python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset trackingnet --threads 15 --num_gpus 5 --neighbor 1 
#python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name ostrack_neighbor_n9
#python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name ostrack_neighbor_n18
#python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name ostrack_ori


#python tracking/analysis_results.py 