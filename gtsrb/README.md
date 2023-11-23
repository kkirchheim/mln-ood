# GTSRB

First, train some networks 

```
python train.py 
```


Run with 
```
python test.py
```



## Ablation Studies 

Run with Shared Parameter Model:  

```
 python test.py --config-name test-shared.yaml n_seeds=1 
```

Number of Rules: 
```
python test.py -m +ablation_n_rules="range(44)" n_seeds=10  use_oe=False use_llm=False paths.models="./models/wrn40/" paths.predictions="./predictions/wrn40/" hydra.launcher.n_jobs=5
```



Vision Transformer:
```
python train.py backbone="vit" paths.models="models/vit-10e/" paths.predictions="predictions/vit-10e/" paths.root="../data/" epochs=10 image_size="[224,224]"
```