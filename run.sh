# dataset=market1501
dataset=PRW_BUformat
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

# batchSize=128
# batchSize=64
batchSize=20
# batchSize=16

# merge_percent=0.07
# merge_percent=0.03
merge_percent=0.050

size_penalty=0.003

python3 run.py --dataset $dataset -b $batchSize --size_penalty $size_penalty -mp $merge_percent 
