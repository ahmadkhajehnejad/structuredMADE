#!/bin/bash
KERAS_BACKEND=tensorflow

#####################################

masks_no=10
hlnum=2
hlsize_list=(25 50 100)
rnd_dim_ord=False
direct_links=False
full_direct_links=False
alg='orig' # 'min_related' # 'Q_restricted' # 'ensemble_Q_restricted_and_orig'
dataset='grid' # 'k_sparse' # 'mnist' # 'Boltzmann' #
rnd_data=False


#####################################

dest_file="config.py"

if [ -f $dest_file ]; then
	rm $dest_file
fi



for hlsize in $hlsize_list; do
	for i in 1 2 3 4 5; do
		echo "num_of_all_masks = $masks_no" >> $dest_file
		echo "num_of_hlayer = $hlnum" >> $dest_file
		echo "hlayer_size = $hlsize" >> $dest_file
		echo "random_dimensions_order = $rnd_dim_ord" >> $dest_file
		echo "direct_links = $direct_links" >> $dest_file
		echo "full_direct_links = $full_direct_links" >> $dest_file
		echo "algorithm = '$alg'" >> $dest_file
		TR=$((200*$i))
		VAL=$((50*$i))
		TE=5000
		echo "train_size = $TR" >> $dest_file
		echo "validation_size = $VAL" >> $dest_file
		echo "test_size = $TE" >> $dest_file
		echo "data_name = '$dataset'" >> $dest_file
		echo "random_data = $rnd_data" >> $dest_file
		cat config_base.py >> $dest_file

		file_str="rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - fulldrlnk:$full_direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no - TR:$TR - VAL:$VAL - TE:$TE"
		cp $dest_file "./results/config -- $file_str.py"
		echo "Running hl_size=$hlsize, i=$i, alg=$alg."
		
		res_path="./results/result -- $file_str.txt"
		srun --gres=gpu:1 python3.5 main.py > $res_path 2> cerr_log

	done
	
done


