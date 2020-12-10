# Unsupervised Person Re-identification for person search
This repository is for UNIST lecture "Machin learning fundamentals" final project. The code is based on the [A Bottom-Up Clustering Approach to Unsupervised Person Re-identification](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) library. 

## Performances
The performances is listed below:

|       | rank-1     | rank-5     | rank-10     | mAP     |
| ---------- | :-----------:  | :-----------: |:-----------:  | :-----------: |
| PRW     | -     | -     |-     | -     |


## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- PRW: You can download datasets here [page](https://drive.google.com/file/d/13-rHAm120Rqhx7oaIB6GJIUB_WiYjK8W/view?usp=sharing) and set the path on run.py line 59(args.data_dir)

## Usage

```shell
sh ./run.sh
```

