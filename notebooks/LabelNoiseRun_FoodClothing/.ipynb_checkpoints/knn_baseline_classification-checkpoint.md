1) Food

Denoise and retrain (finetune baseline 78.61 model): conda activate py2

python resnet_featext.py --batch_size=32 --num_epochs=10 \
--classname_list='/home/krsharma/ClassificationImageText/datasets/Food-101N_release/meta/classnames_list.txt' \
--tr_data_dir='../datasets/Food-101N_release/images' --tr_img_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/knn250_denoised_kv.txt' \
--va_data_dir='../datasets/food-101' --va_img_dir='test' \
--ftext_data_dir='../datasets/Food-101N_release/images' --ftext_img_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/knn250_denoised_kv.txt' \
--model_dir='../trained_models/knn250_round1_food101n_20e' \
--load_model_dir='../trained_models/food101n_20e' --load_model_name='model.pt' \
--run_case='finetune' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=7 --resnet=50

(from scratch instead of finetune)

python resnet_featext.py --batch_size=32 --num_epochs=10 \
--classname_list='/home/krsharma/ClassificationImageText/datasets/Food-101N_release/meta/classnames_list.txt' \
--tr_data_dir='../datasets/Food-101N_release/images' --tr_img_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/knn250_denoised_kv.txt' \
--va_data_dir='../datasets/food-101' --va_img_dir='test' \
--ftext_data_dir='../datasets/Food-101N_release/images' --ftext_img_list='/home/krsharma/ClassificationImageText/trained_models/food101n_20e/knn250_denoised_kv.txt' \
--model_dir='../trained_models/knn250_scratchround1_food101n_20e' \
--run_case='train' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=5 --resnet=50


2) Clothing (The clothing run here finished after 17 hours. 68.83 (baseline before noise removal was 66.64 or 68.94))

python resnet_featext.py --batch_size=32 --num_epochs=10 \
--tr_data_dir='../datasets/clothing1m' --tr_img_list='../trained_models/clothing1m_10e/knn250_denoised_kv_train.txt' \
--va_data_dir='../datasets/clothing1m' --va_img_list='../datasets/clothing1m/clean_test_kv.txt' \
--ftext_data_dir='../datasets/clothing1m' --ftext_img_list='../trained_models/clothing1m_10e/knn250_denoised_kv_train.txt' \
--model_dir='../trained_models/knn250_round1_clothing1m_10e' \
--load_model_dir='../trained_models/clothing1m_10e' --load_model_name='model.pt_epoch_9' \
--run_case='finetune' \
--opt='sgd' --lr=0.001 --momentum=0.9 --gamma=0.1 --stepsize=5 --resnet=50


## Results

num_clean 260009
done preds
3824 4741 3991 4741
P/R/F1 (noise) (0.2693333333333333, 0.2202835332606325, 0.24235152969406118, None)
f1_metrics (macro/unweighted mean) 0.5403696228124817
avg accuracy over classes 0.7336881730935796 AvgErrorRate 0.2663118269064204
-------------
done
num_clean 907465
done preds
4591 7465 6580 7465
P/R/F1 (noise) (0.3525423728813559, 0.10855949895615867, 0.1660015961691939, None)
f1_metrics (macro/unweighted mean) 0.44268211578220684
avg accuracy over classes 0.5668072318642982 AvgErrorRate 0.43319276813570184
-------------
done
92535