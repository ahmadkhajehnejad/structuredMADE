#!/bin/bash
KERAS_BACKEND=tensorflow

#####################################

learn_alpha=False # True # "'heuristic'"
use_best_validated_weights=True
masks_no=10
hlnum=2
hlsize_list=(1200) # (1200) #(800) # 
rnd_dim_ord="'grid_from_center'" # True # "'grid'" # "'1grid-random'" # False #
direct_links="'Full'" # False # True #
algs_list=('min_related') # ('orig') # ('Q_restricted' 'min_related') # ('Q_restricted') # 'ensemble_Q_restricted_and_orig' #
dataset='mnistdps4' # 'mnist' # 'ocrdps1' # 'grid' # 'k_sparse' # 'BayesNet' # 'Boltzmann' #
rnd_data=False # True
fast_train=False


#####################################

dest_file="config.py"

#if [ -f $dest_file ]; then
#rm $dest_file
#fi


for hlsize in ${hlsize_list[@]}; do
        for i in 3; do # 1 2 3 4 5; do
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
                        TR=50000 # $((100*$i))
                        VAL=10000 # $((25*$i))
                        TE=10000
                        echo "train_size = $TR" >> $dest_file
                        echo "validation_size = $VAL" >> $dest_file
                        echo "test_size = $TE" >> $dest_file
                        echo "data_name = '$dataset'" >> $dest_file
                        echo "random_data = $rnd_data" >> $dest_file
                        echo "fast_train = $fast_train" >> $dest_file
                        cat config_base.py >> $dest_file

                        file_str="use_best_validated_weights:${use_best_validated_weights} - learn_alpha:${learn_alpha} - rnddata:${rnd_data} - data:${dataset} - alg:${alg} - drlnk:${direct_links} - rndord:${rnd_dim_ord} - hlnum:${hlnum} - hlsize:${hlsize} - masksNum:${masks_no} - fastTrain:${fast_train} - TR:${TR} - VAL:${VAL} - TE:${TE}"
                        cp ${dest_file} "./results_revision/config -- ${file_str}.py"
                        echo "Running hl_size=$hlsize, i=$i, alg=$alg."

                        res_path="./results_revision/result -- ${file_str}.txt"
                        python -u main.py > "${res_path}" 2> cerr_log

                done
        done

done

