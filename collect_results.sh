#!/bin/bash
ALG='Q_restricted'
FILE_NAME_PREFIX='results_mnist_6'


for HL in 'numhl2_hlsize300' 'numhl2_hlsize600'
do
	for prefix in 'Average NLLs: ' 'avg Num of Connections '
	do
		results="$HL $prefix ["
		for i in 1 2 3 4
		do
			DEST_PATH="./$ALG/$HL/$i"
			TR=$((200*$i))
			VAL=$((50*$i))
			TE=5000
			catched_line=$(cat $DEST_PATH/$FILE_NAME_PREFIX"_tr"$TR"_val"$VAL"_te"$TE".txt" | grep "$prefix")
			new_result=${catched_line#$prefix}
			if [ $((i)) -lt 4 ]; then
				results="$results $new_result,"
			else
				results="$results $new_result]"
			fi
		done
		echo $results
	done
done
#echo "============="
#
#for i in 1 2 3 4
#do
#	DEST_PATH="./results/$DATA_NAME/$ALG/$HL/$i"
#	TR=$((200*$i))
#	VAL=$((50*$i))
#	cat $DEST_PATH/$FILE_NAME_PREFIX"_tr"$TR"_val"$VAL"_te"$TE".txt" | grep "Variance num of"
#done
