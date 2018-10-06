#!/bin/bash
KERAS_BACKEND=tensorflow

#####################################

masks_no=10
hlnum=2
hlsize_list=(25 50 100 200 300)
rnd_dim_ord=True # False # 'grid'
direct_links=True # False # 'Full'
algs_list=("orig" "Q_restricted") # 'min_related' # 'ensemble_Q_restricted_and_orig'
dataset='grid' # 'k_sparse' # 'mnist' # 'Boltzmann' #
rnd_data=False
prefix='Average KLs: ' # 'avg Num of Connections ' # 'Variance KLs: ' # 


#####################################

for hlsize in ${hlsize_list[@]}; do
	for alg in ${algs_list[@]}; do
		file_str_1="rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no"
		results="$file_str_1 - $prefix ["
		for i in 1 2 3 4 5; do
			TR=$((200*$i))
			VAL=$((50*$i))
			TE=5000
			file_str="$file_str_1 - TR:$TR - VAL:$VAL - TE:$TE"
			res_path="./results/result -- $file_str.txt"
			catched_line=$(cat "$res_path" | grep "$prefix")
			new_result=${catched_line#$prefix}
			if [ $((i)) -lt 5 ]; then
				results="$results $new_result,"
			else
				results="$results $new_result]"
			fi

		done
		echo $results
	done
done

