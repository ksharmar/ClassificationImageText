# Hymenoptera dataset runs 
python resnet_featext.py --batch_size=64 --num_epochs=5 --data_dir='../datasets/hymenoptera_data' --img_dir='train' --model_dir='../trained_models/hymenoptera_last' --opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=7 --resnet=50

# Food dataset runs (feature extraction, avg, cleannet, repeat once)
     
A) train acc = models saved at each epoch

python resnet_featext.py --run_case='train' --batch_size=64 --num_epochs=20 \
--tr_data_dir='../datasets/Food-101N_release' --tr_img_dir='images' \
--va_data_dir='../datasets/food-101' --va_img_dir='test' \
--model_dir='../trained_models/food101n_20e' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=7 --resnet=50

python resnet_featext.py --run_case='train' --batch_size=64 --num_epochs=10 \
--tr_data_dir='../datasets/clothing1m' --tr_img_list='../datasets/clothing1m/noisy_train_kv.txt' \
--va_data_dir='../datasets/clothing1m' --va_img_list='../datasets/clothing1m/clean_test_kv.txt' \
--model_dir='../trained_models/clothing1m_10e' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=5 --resnet=50

B) val acc = result to select a model

python resnet_test.py --batch_size=64 \
--data_dir='../datasets/food-101' --img_dir='test' \
--model_dir='../trained_models/food101n' --model_path='model.pt_epoch_9'
> Food test - train=93 and test=78.61 (should be 81 or 84 depending on learning rate) 

python resnet_featext.py --run_case='val' --batch_size=32 \
--tr_data_dir='../datasets/clothing1m' --tr_img_list='../datasets/clothing1m/noisy_train_kv.txt' \
--va_data_dir='../datasets/clothing1m' --va_img_list='../datasets/clothing1m/clean_test_kv.txt' \
--model_dir='../trained_models/clothing1m_10e' --saved_model_name='model.pt_epoch_9'
> Clothing test - train=99 and test=66.64 (should be 68.94)



C) feat ext = extract features at the model

python resnet_featext.py --run_case='feat_ext' --batch_size=64 \
--tr_data_dir='../datasets/Food-101N_release' --tr_img_dir='images' \
--va_data_dir='../datasets/food-101' --va_img_dir='test' \
--model_dir='../trained_models/food101n' --saved_model_name='model.pt_epoch_9'


python resnet_featext.py --run_case='feat_ext' --batch_size=64 \
--tr_data_dir='../datasets/clothing1m' --tr_img_list='../datasets/clothing1m/extra_verified_kv.txt' \
--va_data_dir='../datasets/clothing1m' --va_img_list='../datasets/clothing1m/clean_test_kv.txt' \
--model_dir='../trained_models/clothing1m_10e' --saved_model_name='model.pt_epoch_9'


# Clean-net baseline

## Step 1:

First create files using format_for_cleannet notebook. Then run following to get input files to run cleannet.

python util/convert_data.py --split=val \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_val.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/'

python util/convert_data.py --split=val \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_val.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/'

python util/convert_data.py --split=val \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_val.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/'

> found 3824/4741 positive samples

> found 4591/7465 positive samples (clothing)


python util/convert_data.py --split=train \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_train.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/'

python util/convert_data.py --split=train \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_train.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/'

python util/convert_data.py --split=train \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_train.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/'

> found 43121/52K positive samples

> found 15210/24637 positive samples

python util/convert_data.py --split=all \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_all.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/'

python util/convert_data.py --split=all \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_all.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/'

python util/convert_data.py --split=all \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--data_path='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_all.tsv' \
--output_dir='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/'


## Step 2 find reference: 
python util/find_reference.py \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/all.npy' \
--output_dir='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/' \
--num_ref=50 --img_dim=2048

python util/find_reference.py \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/all.npy' \
--output_dir='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/' \
--num_ref=50 --img_dim=2048

python util/find_reference.py \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/all.npy' \
--output_dir='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/' \
--num_ref=50 --img_dim=2048


## Step 3 train cleannet:
python train.py \
--data_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/checkpoints/" \
--log_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/log/" \
--num_ref=50 --img_dim 2048 --
--n_step=60000 --lr_update=30000 --val_interval=2000 \
--batch_size_sup=32 --batch_size_unsup=32 --val_batch_size 64 \
--val_sim_thres=0.1 --neg_weight=5.0 --embed_norm='log' \
--learning_rate=0.01 --lr_decay=0.1 --dropout_rate=0.2 --weight_decay=0.0001 --momentum=0.9

python train.py \
--data_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/checkpoints/" \
--log_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/log/" \
--num_ref=50 --img_dim 2048 --
--n_step=60000 --lr_update=30000 --val_interval=2000 \
--batch_size_sup=32 --batch_size_unsup=32 --val_batch_size 64 \
--val_sim_thres=0.1 --neg_weight=2.5 --embed_norm='log' \
--learning_rate=0.01 --lr_decay=0.1 --dropout_rate=0.2 --weight_decay=0.0001 --momentum=0.9

python train.py \
--data_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/checkpoints/" \
--log_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/log/" \
--num_ref=50 --img_dim 2048 --
--n_step=60000 --lr_update=30000 --val_interval=2000 \
--batch_size_sup=32 --batch_size_unsup=32 --val_batch_size 64 \
--val_sim_thres=0.1 --neg_weight=5.0 --embed_norm='log' \
--learning_rate=0.01 --lr_decay=0.1 --dropout_rate=0.2 --weight_decay=0.0001 --momentum=0.9


## Step 4 inference (run validation) cleannet [predict and score val samples at different thresholds]:
python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/cleannet_predval.txt" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/checkpoints/" \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=val --val_sim_thres=0.1 --embed_norm='log'


python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/cleannet_predval.txt" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/checkpoints/" \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=val --val_sim_thres=0.1 --embed_norm='log'

python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/cleannet_predval.txt" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/checkpoints/" \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=val --val_sim_thres=0.1 --embed_norm='log'


## Average baseline

python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/avgbaseline_predval.txt" \
--checkpoint_dir=None \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=avgbaseline --val_sim_thres=0.1 --embed_norm='log'

python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/clothing1m_10e/cleannet/avgbaseline_predval.txt" \
--checkpoint_dir=None \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=avgbaseline --val_sim_thres=0.1 --embed_norm='log'

python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/" \
--class_list='/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/round1_food101n_20e/cleannet/avgbaseline_predval.txt" \
--checkpoint_dir=None \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=avgbaseline --val_sim_thres=0.1 --embed_norm='log'


# ROUND 1: Denoise and re-train

## inference (run prediction) cleannet [for samples without verification label]:
    
python inference.py \
--data_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/" \
--image_feature_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_all.tsv' \
--class_list='/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet_classnames.txt' \
--output_file="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/cleannet_predtrain.txt" \
--checkpoint_dir="/home/krsharma/labelnoise/trained_models/food101n_20e/cleannet/checkpoints/" \
--img_dim=2048 --num_ref=50 --batch_size=64 \
--mode=inference --embed_norm='log'


## denoise and retrain feat extraction

python resnet_featext.py --run_case='train' --batch_size=32 --num_epochs=15 \
--classname_list='/home/krsharma/labelnoise/datasets/Food-101N_release/meta/classnames_list.txt' \
--tr_data_dir='../datasets/Food-101N_release/images' --tr_img_list='/home/krsharma/labelnoise/trained_models/food101n_20e/denoised_kv.txt' \
--va_data_dir='../datasets/food-101' --va_img_dir='test' \
--ftext_data_dir='../datasets/Food-101N_release' --ftext_img_dir='images' \
--model_dir='../trained_models/round1_food101n_20e' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=7 --resnet=50


    


