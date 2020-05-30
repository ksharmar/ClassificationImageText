# testing: inside RobustNoiseRank run. Run robust noiserank with class prototypes selected as kmeans cluster centroids

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/cleannet_all.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/ref.tsv' \
--list_k 10 30 50 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/'

# 1) run cleanent prototype selection on 256d (env tf16) 

python util/convert_data.py --split=all \
--class_list=''/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/cleannet_classnames.txt'' \
--data_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/cleannet_all_256d.tsv' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/'


python util/find_reference.py \
--class_list='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/all.npy' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/' \
--num_ref=50 --img_dim=256


# 2) first format the ref numpy to tsv, then inside RobustNoiseRank run (env faiss)

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/ref_100.tsv' \
--list_k 50 100 250 500 1000 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/results_100/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/ref_500.tsv' \
--list_k 250 500 1000 1500 2500 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/results_500/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/ref_190.tsv' \
--list_k 50 100 250 500 1000 1500 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/results_190/'


# 3) while evaluating (evaluate accuracy on 14K validation set => actually 7K val only). 

'/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/results_100/k=100_a=0.5_bf=1.0_b=1.0_e=2.0.txt'

result: P/R/F1 (noise) (0.6354902514834699, 0.7825330549756437, 0.7013878060190238, None)
f1_metrics (macro/unweighted mean) 0.738271688614772
avg accuracy over classes 0.7441506457285033 AvgErrorRate 0.2558493542714967


# 4) then provide detected noise images paths (40% detected as noise) so that finetune on denoised subset.

(detected noise ids are from 0 to XX of cleannet_all_256d.tsv - Row number)


#### CLOTHING round 1 

python util/convert_data.py --split=all \
--class_list=''/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/cleannet_classnames.txt'' \
--data_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/cleannet_all_256d.tsv' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/'

python util/find_reference.py \
--class_list='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/all.npy' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/' \
--num_ref=100 --img_dim=256

python util/find_reference.py \
--class_list='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/cleannet_classnames.txt' \
--input_npy='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/all.npy' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/' \
--num_ref=200 --img_dim=256


>>prototype count (100 (arbit) or 40=meanrulethumb, here 100 (minthumbrule) or 190=meanrulethumb)

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/ref_100.tsv' \
--list_k 10 20 50 100 250 500 1000 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/results_100/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/ref_200.tsv' \
--list_k 10 20 50 100 250 500 1000 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/results_200/'

##### FOOD dataset

>> d_00000_781250x256.pickle  
eluo_clothing1m_everstore_handle_with_label_and_type.csv


python util/convert_data.py --split=all \
--class_list=''/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_classnames.txt' \
--data_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/'

python util/find_reference.py \
--class_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_classnames.txt' \
--input_npy='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/all.npy' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/' \
--num_ref=40 --img_dim=256


>> all.npy                
cleannet_all_256d.tsv


python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_30.tsv' \
--list_k 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_30/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_40.tsv' \
--list_k 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_40/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_50.tsv' \
--list_k 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_50/'


>> 1400 (50 100 250 500 1000) clothing
3020-5050 (10 20 30 50 100 250 500 1000) food

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_30.tsv' \
--list_k 10 20 30 50 100 250 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_30/'


python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_40.tsv' \
--list_k 10 20 30 50 100 250 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_40/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_50.tsv' \
--list_k 10 20 30 50 100 250 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_50/'



python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/ref_100.tsv' \
--list_k 5 10 15 20 30 50 100 250 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_emb/results_100/'



#### Food round 1

python util/convert_data.py --split=all \
--class_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_classnames.txt' \
--data_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_all_256d.tsv' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/'

python util/find_reference.py \
--class_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_classnames.txt' \
--input_npy='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/all.npy' \
--output_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/' \
--num_ref=40 --img_dim=256


python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/ref_40.tsv' \
--list_k 1 2 3 4 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 1 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/results_40_e1/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/ref_50.tsv' \
--list_k 1 2 3 4 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/results_50/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/ref_10.tsv' \
--list_k 1 2 3 4 5 10 15 20 30 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/results_10/'


python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/cleannet_all_256d.tsv' \
--list_k 5 10 50 250 500 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 1 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/robustnoiserank/food-256d_round1/results_noiserank/'

#### Results

clothing 

**ROUND0 = 247021 detected = np.loadtxt('/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_emb/results_190/k=50_a=0.5_bf=1.0_b=1.0_e=2.0.txt')
P/R/F1 (noise) (0.7011772853185596, 0.7045929018789144, 0.7028809441166263, None)
f1_metrics (macro/unweighted mean) 0.7580722347110183
avg accuracy over classes 0.7645810177532378 AvgErrorRate 0.23541898224676217

**ROUND1=results_200_K=50_P/R/F1 (noise) (0.6944625407166124, 0.7418232428670842, 0.7173620457604307, None)
f1_metrics (macro/unweighted mean) 0.7652022781662158
avg accuracy over classes 0.7718288428116713 AvgErrorRate 0.2281711571883287

food

**131547: 50-5-P/R/F1 (noise) (0.4117946110828673, 0.8833151581243184, 0.5617198335644938, None)
f1_metrics (macro/unweighted mean) 0.6850733147816408
avg accuracy over classes 0.7328569955055766 AvgErrorRate 0.2671430044944234


**ROUND1= 40-5-P/R/F1 (noise) (0.40156326331216413, 0.8964013086150491, 0.5546558704453441, None)
f1_metrics (macro/unweighted mean) 0.676069880604691
avg accuracy over classes 0.7211837711990654 AvgErrorRate 0.2788162288009346


# analysis clothing

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/cleannet_all_256d.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/ref_200.tsv' \
--list_k 50 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/analysis_200/'

python main.py \
--input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/test.tsv' \
--ref_input_path='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/ref_200.tsv' \
--list_k 2 \
--list_alpha 0.5 --list_bf 1.0 \
--list_b 1 --list_e 2 \
--save_path_dir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/clothing-256d_round1/analysis_test/'