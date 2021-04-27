ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

python -u $ROOT/tools/valid_siammask.py \
    --config=config.json -b 4 \
    -j 6 \
    --epochs 20 \
    --log log.txt \
    2>&1 | tee valid.log
