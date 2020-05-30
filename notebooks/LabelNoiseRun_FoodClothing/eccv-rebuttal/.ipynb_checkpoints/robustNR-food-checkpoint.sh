datadir='/home/krsharma/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank/eccvR_clothing-256d_emb'
codedir='/home/krsharma/ClassificationImageText/clean-net'
outdir='./'

python $codedir/util/convert_data.py --split=all \
--class_list=$datadir/cleannet_classnames.txt \
--data_path=$datadir/cleannet_all_256d.tsv \
--output_dir=$datadir

for num in 190; do  # food - 40 30 50 100
    mkdir $datadir/ref$num
    
    python $codedir/util/find_reference.py \
    --class_list=$datadir/cleannet_classnames.txt \
    --input_npy=$datadir/all.npy \
    --output_dir=$datadir/ref$num \
    --num_ref=$num --img_dim=256
    
    mv $datadir/ref$num/ref.npy $datadir/ref_$num.npy
done