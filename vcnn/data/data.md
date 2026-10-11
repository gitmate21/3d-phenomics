# Dataset format

The data is organized as a PyTables `.h5` file. Each file contains a group
`/process/` with four arrays:

| array | shape | dtype | meaning |
|-------|-------|-------|---------|
| `x`    | `(N, 3)` | float32 | voxel coordinates of all samples (concatenated) |
| `y1`   | `(N,)`   | int     | **classification** label — semantic segmentation (`0` = background/noise, `1` = plant) |
| `y2`   | `(N,)`   | int     | **segmentation** label — instance segmentation (`0..K-1`, one id per leaf/stem instance) |
| `idxs` | `(M, 2)` | int     | `(start, end)` slice bounds delimiting each of the `M` samples/plants |

The bundled demo file is `data/x_y1_y2_idx_33_withoutnoise.h5` (33 plants).

## Loading (quick check)

```python
from scripts.h5 import h5f_
with h5f_("data/x_y1_y2_idx_33_withoutnoise.h5", "r") as f:
    x    = f.get_node("/process/x").read()
    y1   = f.get_node("/process/y1").read()
    y2   = f.get_node("/process/y2").read()
    idxs = f.get_node("/process/idxs").read()
print(x.shape, y1.shape, y2.shape, idxs.shape)
```
