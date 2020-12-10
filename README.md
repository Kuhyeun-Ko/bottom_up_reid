# Unsupervised Person Re-identification for Person Search
This repository is for UNIST lecture "Machin learning fundamentals" final project. The code is based on the [A Bottom-Up Clustering Approach to Unsupervised Person Re-identification](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) library. 

## Performances
The performances is listed below:

|       | mAP     |rank-1     | rank-5     | rank-10     | 
| ---------- | :-----------:  | :-----------: |:-----------:  | :-----------: |
| PRW     |  19.1%| 47.3%| 59.2% | 63.7% |
| PRW with constraint    | 19.4% | 49.4%|59.6%| 64.9%|

## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- PRW: You can download datasets [here](https://drive.google.com/file/d/13-rHAm120Rqhx7oaIB6GJIUB_WiYjK8W/view?usp=sharing) and set the datasets path on run.py line 59(args.data_dir).

## Usage

```shell
sh ./run.sh
```

