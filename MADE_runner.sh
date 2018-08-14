#!/bin/bash
KERAS_BACKEND=tensorflow


for HL in 'numhl2_hlsize100' 'numhl2_hlsize50' 'numhl2_hlsize25'  
do
	for i in 1 2 3 4 5
	do
		for ALG in 'ensemble_Q_restricted_and_orig' #'min_related' 'orig'
		do
			for NM in 'num_masks_20'
			do
				echo "$HL Running case #$i of 5 for method $ALG ..."
				DEST_PATH="./results/grid_new/$ALG/$NM/$HL/$i"
				TR=$((200*$i))
				VAL=$((50*$i))
				cp $DEST_PATH/config.py .
				srun --gres=gpu:1 python3.5 main.py > "$DEST_PATH/results_grid_4by4_tr"$TR"_val"$VAL"_teFULL.txt" 2> cerr_log
			done
		done
	done
done	
