use_best_validated_weights=True
learn_alpha=False
masks_no=10
hlnum=2
hlsize_list=(200)
rnd_dim_ord=True # False # 'grid'
direct_links="'Full'" # True # False #
algs_list=("orig" "Q_restricted") # ('min_related' "random_Q_restricted") # 'ensemble_Q_restricted_and_orig'
dataset='k_sparse' # 'grid' # 'mnist' # 'Boltzmann' #
rnd_data=False
prefix='Average KLs: ' # 'Variance KLs: ' # 'avg Num of Connections ' #


#####################################

for hlsize in ${hlsize_list[@]}; do
        for alg in ${algs_list[@]}; do
                file_str_1="use_best_validated_weights:$use_best_validated_weights - learn_alpha:$learn_alpha - rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no"
                #file_str_1="learn_alpha:$learn_alpha - rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no"
                results="avgKL__rndord_${alg}__hlsize_$hlsize =  ["
                for i in 1 2 3 4 5; do
                        TR=$((100*$i))
                        VAL=$((25*$i))
                        TE=5000
                        file_str="$file_str_1 - TR:$TR - VAL:$VAL - TE:$TE"
                        res_path="./results_good_early_stop/result -- $file_str.txt"
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

