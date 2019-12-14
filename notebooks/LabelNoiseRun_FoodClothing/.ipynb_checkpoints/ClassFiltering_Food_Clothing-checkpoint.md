## FOOD

# create imglist file from cleannet_val.tsv

import pandas as pd
df = pd.read_csv('cleannet_val.tsv', sep='\t', header=None)
df[[1,2]].to_csv('cleannet_val_logits_imlist.tsv', sep=' ', header=None, index=False)


# extract logits (for cleannet_val.tsv) from resnet

python resnet_featext.py --run_case='logits_ext' --batch_size=64 \
--classname_list='../trained_models/food101n_20e/cleannet_classnames.txt' \
--ftext_data_dir='../datasets/Food-101N_release/images' --ftext_img_list='../trained_models/food101n_20e/cleannet_val_logits_imlist.tsv' \
--model_dir='../trained_models/food101n_20e/' --saved_model_name='model.pt' \
--tr_data_dir='../datasets/food-101' --tr_img_dir='test' \
--va_data_dir='../datasets/food-101' --va_img_dir='test'

# run class filtering

python classfiltering.py \
--val_logits_file='../trained_models/food101n_20e/val_logits.txt' \
--val_tsv='../trained_models/food101n_20e/cleannet_val.tsv'


-------------starting for k=2-----------------
considered classes (by k filtering) (4741, 2)
P/R/F1 (noise) (0.765079365079365, 0.5256270447110142, 0.6231415643180349, None)
f1_metrics (macro/unweighted mean) 0.7748348023228486
avg accuracy over classes 0.8772877510485675 AvgErrorRate 0.12271224895143251




## CLOTHING

# create imglist file from cleannet_val.tsv

import pandas as pd
df = pd.read_csv('cleannet_val.tsv', sep='\t', header=None)
df[[1,2]].to_csv('cleannet_val_logits_imlist.tsv', sep=' ', header=None, index=False)


# extract logits (for cleannet_val.tsv) from resnet

python resnet_featext.py --run_case='logits_ext' --batch_size=64 \
--classname_list='../trained_models/clothing1m_10e/cleannet_classnames.txt' \
--ftext_data_dir='../datasets/clothing1m' --ftext_img_list='../trained_models/clothing1m_10e/cleannet_val_logits_imlist.tsv' \
--model_dir='../trained_models/clothing1m_10e/' --saved_model_name='model.pt' \
--tr_data_dir='../datasets/clothing1m' --tr_img_list='../datasets/clothing1m/noisy_train_kv.txt' \
--va_data_dir='../datasets/clothing1m' --va_img_list='../datasets/clothing1m/clean_test_kv.txt'

# run class filtering

python classfiltering.py \
--val_logits_file='../trained_models/clothing1m_10e/val_logits.txt' \
--val_tsv='../trained_models/clothing1m_10e/cleannet_val.tsv'

-------------starting for k=1-----------------
considered classes (by k filtering) (7465, 1)
P/R/F1 (noise) (0.759958613554061, 0.511134307585247, 0.6111920116496775, None)
f1_metrics (macro/unweighted mean) 0.7132814745594036
avg accuracy over classes 0.7418822039937857 AvgErrorRate 0.2581177960062143