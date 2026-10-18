# Data layout

`scripts/prepare_pheno4d.py` turns raw Pheno4D (`<Plant>/<Plant>_<day>_a.txt`, columns `x y z label ...`) into:

```
<data root>/
  Tomato01/Tomato01_03.npz     # xyz float32 (N,3), label uint8 (N,): 0 soil, 1 stem, 2 leaf
  ...
  splits/Pheno4D_{train,val,trainval,test}.txt   # relative npz paths, split by plant
```

Soil points are ignored in the loss; the network predicts stem (0) / leaf (1).
