#!/bin/bash
KERAS_BACKEND=tensorflow

#####################################

learn_alpha=False # True # "'heuristic'"
use_best_validated_weights=True
masks_no=10
hlnum=6
hlsize_list=(150 300 600) #
rnd_dim_ord=True # "'grid'" # "'grid_from_center'" # "'1grid-random'" # False #
direct_links="'Full'" # False # True #
algs_list=('orig' 'Q_restricted') # ('min_related') # ('Q_restricted') # 'ensemble_Q_restricted_and_orig' #
dataset='BayesNet' # 'mnistdps4' # 'ocrdps2' # 'k_sparse' # 'grid' # 'Boltzmann' #
rnd_data=False # True


#####################################

dest_file="config.py"

#if [ -f $dest_file ]; then
#rm $dest_file
#fi


for hlsize in ${hlsize_list[@]}; do
        for i in 1 2 3 4 5 6 7 8 9 10; do
                for alg in ${algs_list[@]}; do
                        echo ' ' > $dest_file
                        echo "learn_alpha = $learn_alpha" >> $dest_file
                        echo "use_best_validated_weights = $use_best_validated_weights" >> $dest_file
                        echo "num_of_all_masks = $masks_no" >> $dest_file
                        echo "num_of_hlayer = $hlnum" >> $dest_file
                        echo "hlayer_size = $hlsize" >> $dest_file
                        echo "random_dimensions_order = $rnd_dim_ord" >> $dest_file
                        echo "direct_links = $direct_links" >> $dest_file
                        echo "algorithm = '$alg'" >> $dest_file
                        TR=$((100*$i))
                        VAL=$((25*$i))
                        TE=5000
                        echo "train_size = $TR" >> $dest_file
                        echo "validation_size = $VAL" >> $dest_file
                        echo "test_size = $TE" >> $dest_file
                        echo "data_name = '$dataset'" >> $dest_file
                        echo "random_data = $rnd_data" >> $dest_file
                        cat config_base.py >> $dest_file

                        file_str="use_best_validated_weights:$use_best_validated_weights - learn_alpha:$learn_alpha - rnddata:$rnd_data - data:$dataset - alg:$alg - drlnk:$direct_links - rndord:$rnd_dim_ord - hlnum:$hlnum - hlsize:$hlsize - masksNum:$masks_no - TR:$TR - VAL:$VAL - TE:$TE"
                        cp $dest_file "./results/config -- $file_str.py"
                        echo "Running hl_size=$hlsize, i=$i, alg=$alg."

                        res_path="./results/result -- $file_str.txt"
                        python main.py > $res_path 2> cerr_log

                done
        done

done

