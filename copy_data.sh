mkdir /scratch/cliao25
mkdir /scratch/cliao25/data
cd /scratch/cliao25/data
cp -r ~/siammask/data/*.py ~/siammask/data/*.sh .

echo copying coco
cp -r ~/siammask/data/coco /scratch/cliao25/data/coco
cd /scratch/cliao25/data/coco
echo untaring coco
tar -xf crop511.tar
echo removing coco.tar
rm -f crop511.tar

echo copying vid
cp -r ~/siammask/data/vid /scratch/cliao25/data/vid
cd /scratch/cliao25/data/vid
echo untaring vid
tar -xf crop511.tar
echo removing vid.tar
rm -f crop511.tar

echo copying yte vos
cp -r ~/siammask/data/ytb_vos /scratch/cliao25/data/ytb_vos
cd /scratch/cliao25/data/ytb_vos
echo untaring youtube vos
unzip -q crop511.zip
echo removing youtube vos
rm -f crop511.zip
