# ClassificationImageText 

## Implementations

- Clean-net (label noise)
- Code for Bert (text feature extraction, classification, tweet extraction from tweepy)
- Code for Resnet50 (image feature extraction, classification)

## Environments

- cleannet/avgbaseline
```tf16: conda install -c conda-forge tensorflow=1.6.0```

- bertfinetuning
```bert: pytorch env```

- noiserank
```faiss: conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9```

- python2 loading older models
```
py2: conda create --name py2 python=2.7 (install matplotlib and pandas again)
# CUDA 9.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
```

- general
```
conda env remove -n faiss
conda create -n faiss python=3.6 
python -m ipykernel install --user --name faiss
```

## Datasets
- waseem dataset (label_map = {'racism':0, 'sexism':1, 'both':2, 'neither':3})


## Experiments ECCV (NoiseRank with Prototypes)
- /home/krsharma/workspace/ClassificationImageText/trained_models/food101n_20e/robustnoiserank
- /home/krsharma/workspace/ClassificationImageText/trained_models/clothing1m_10e/robustnoiserank
