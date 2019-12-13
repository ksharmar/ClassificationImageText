cleannet/avgbaseline
tf16: conda install -c conda-forge tensorflow=1.6.0

bertfinetuning
bert: pytorch env

noiserank
faiss: conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
https://github.com/facebookresearch/faiss/blob/master/INSTALL.md?fbclid=IwAR2BOUltKoVuaSPTsJ_s5RxA9JmrgbKYAhDO1CIf-rHi84SgpzawOSd0z8I

conda env remove -n faiss
conda create -n faiss python=3.6 
python -m ipykernel install --user --name faiss

label_map = {'racism':0, 'sexism':1, 'both':2, 'neither':3}

