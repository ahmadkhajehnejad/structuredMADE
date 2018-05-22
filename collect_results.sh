#!/bin/bash
ALG='orig'
HL='numhl2_hlsize24'
DATA_NAME='grid_pos'
FILE_NAME_PREFIX='results_grid_4by4'

for i in 1 2 3 4 5
do
	DEST_PATH="./results/$DATA_NAME/$ALG/$HL/$i"
	TR=$((200*$i))
	VAL=$((50*$i))
	cat $DEST_PATH/$FILE_NAME_PREFIX"_tr"$TR"_val"$VAL"_teFULL.txt" | grep "Average KLs"
done

echo "============="

for i in 1 2 3 4 5
do
	DEST_PATH="./results/$DATA_NAME/$ALG/$HL/$i"
	TR=$((200*$i))
	VAL=$((50*$i))
	cat $DEST_PATH/$FILE_NAME_PREFIX"_tr"$TR"_val"$VAL"_teFULL.txt" | grep "Variance KLs"
done
