#!/bin/bash
KERAS_BACKEND=tensorflow


for HL in 'numhl2_hlsize80' 'numhl2_hlsize30'
do
	for i in 1 2 3 4 5
	do
		for ALG in 'orig' 'Q_restricted' #'min_related'
		do
			echo "$HL Running case #$i of 5 for method $ALG ..."
			DEST_PATH="./results/Boltzmann/$ALG/$HL/$i"
			TR=$((200*$i))
			VAL=$((50*$i))
			cp $DEST_PATH/config.py .
			srun --gres=gpu:1 python3.5 main.py > "$DEST_PATH/results_Boltzmann_10&10_tr"$TR"_val"$VAL"_teFULL.txt" 2> cerr_log
		done
	done
done	
