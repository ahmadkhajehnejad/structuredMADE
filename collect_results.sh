use_best_validated_weights=True
learn_alpha=False
masks_no=10
hlnum=2
hlsize_list=(25 50 100)
rnd_dim_ord_list=(False True "'grid'") # False # 
direct_links="'Full'" # True # False #
algs_list=("orig" 'min_related' "Q_restricted") # ("random_Q_restricted") # 'ensemble_Q_restricted_and_orig'
dataset='grid' # 'k_sparse' # 'mnist' # 'Boltzmann' #
rnd_data=False
fast_train=False
prefix_avg='Average KLs: ' # 'Variance KLs: ' # 'avg Num of Connections ' #
prefix_var='Variance KLs: ' # 'Variance KLs: ' # 'avg Num of Connections ' #


#####################################

for rnd_dim_ord in ${rnd_dim_ord_list[@]}; do
for hlsize in ${hlsize_list[@]}; do
        for alg in ${algs_list[@]}; do
                file_str_1="use_best_validated_weights:$use_best_validated_weights - learn_alpha:$learn_alpha - rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no - fastTrain:$fast_train"
                #file_str_1="learn_alpha:$learn_alpha - rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no"
		echo " "
                echo "order: $rnd_dim_ord  ,  hlsize: $hlsize,  algo: $alg  ###########"
                for i in 1 2 3 4 5; do
                        TR=$((100*$i))
                        VAL=$((25*$i))
                        TE=5000
                        file_str="$file_str_1 - TR:$TR - VAL:$VAL - TE:$TE"
                        res_path="./results_revision_biasedMADE/result -- $file_str.txt"
                        catched_line_avg=$(cat "$res_path" | grep "$prefix_avg")
                        catched_line_var=$(cat "$res_path" | grep "$prefix_var")
                        avg=${catched_line_avg#$prefix_avg}
                        var=${catched_line_var#$prefix_var}
			std=$(echo "sqrt($var)" | bc)
			echo "($TR, $avg) +- ($var, $var)"
                done
        done
done
done
