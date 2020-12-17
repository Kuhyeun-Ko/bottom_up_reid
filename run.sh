dataset=PRW_BUformat

# batchSize=128
# batchSize=64
# batchSize=20
batchSize=16

merge_percent=0.050

size_penalty=0.003

# 4 options
# bucc: bottom_up_clustering_constraint
# burn: bottom_up_real_negative
# mst: ms_table
# msrn: ms_real_negative

bucc=False
burn=False
mst=False
msrn=False

python3 run.py --dataset $dataset -b $batchSize --size_penalty $size_penalty -mp $merge_percent --bucc $bucc --burn $burn --mst $mst --msrn $msrn
