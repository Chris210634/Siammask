export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

dataset=VOT2018

bash test.sh checkpoint_e40.pth $dataset 2 > test_dummy.txt
