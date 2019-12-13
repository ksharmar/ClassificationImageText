# ClassificationImageText 

## Implementations

- Clean-net (label noise)
- Code for Bert (text feature extraction, classification, tweet extraction from tweepy)
- Code for Resnet50 (image feature extraction, classification)

## Todo

- data syn noise add (uni, nonuni, diff noise rates)
- ours (iterative, checkL2norm on 768d vectors)
- Implement and run DRAE and class filtering on Food, Clothing, Text (Waseem, Synthetic) datasets

## Environments

- cleannet/avgbaseline
```tf16: conda install -c conda-forge tensorflow=1.6.0```

- bertfinetuning
```bert: pytorch env```

- noiserank
```faiss: conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9```

- general
```
conda env remove -n faiss
conda create -n faiss python=3.6 
python -m ipykernel install --user --name faiss
```

## Datasets
- waseem dataset (label_map = {'racism':0, 'sexism':1, 'both':2, 'neither':3})
