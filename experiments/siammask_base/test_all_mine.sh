export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

dataset=VOT2018

bash test.sh snapshot/checkpoint_e1.pth $dataset 1 > test_e1.txt
bash test.sh snapshot/checkpoint_e2.pth $dataset 1 > test_e2.txt
bash test.sh snapshot/checkpoint_e3.pth $dataset 1 > test_e3.txt
bash test.sh snapshot/checkpoint_e4.pth $dataset 1 > test_e4.txt
bash test.sh snapshot/checkpoint_e5.pth $dataset 1 > test_e5.txt
bash test.sh snapshot/checkpoint_e6.pth $dataset 1 > test_e6.txt
bash test.sh snapshot/checkpoint_e7.pth $dataset 1 > test_e7.txt
bash test.sh snapshot/checkpoint_e8.pth $dataset 1 > test_e8.txt
bash test.sh snapshot/checkpoint_e9.pth $dataset 1 > test_e9.txt
bash test.sh snapshot/checkpoint_e10.pth $dataset 1 > test_e10.txt
bash test.sh snapshot/checkpoint_e11.pth $dataset 1 > test_e11.txt
bash test.sh snapshot/checkpoint_e12.pth $dataset 1 > test_e12.txt
bash test.sh snapshot/checkpoint_e13.pth $dataset 1 > test_e13.txt
bash test.sh snapshot/checkpoint_e14.pth $dataset 1 > test_e14.txt
bash test.sh snapshot/checkpoint_e15.pth $dataset 1 > test_e15.txt
bash test.sh snapshot/checkpoint_e16.pth $dataset 1 > test_e16.txt
bash test.sh snapshot/checkpoint_e17.pth $dataset 1 > test_e17.txt
bash test.sh snapshot/checkpoint_e18.pth $dataset 1 > test_e18.txt
bash test.sh snapshot/checkpoint_e19.pth $dataset 1 > test_e19.txt
bash test.sh snapshot/checkpoint_e20.pth $dataset 1 > test_e20.txt
