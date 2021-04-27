ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e1.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e2.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e3.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e4.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e5.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e6.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e7.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e8.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e9.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e11.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e12.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e13.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e14.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e15.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e16.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e17.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e18.pth  DAVIS 0
bash test_mask_refine.sh config_davis.json snapshot/checkpoint_e19.pth  DAVIS 0
