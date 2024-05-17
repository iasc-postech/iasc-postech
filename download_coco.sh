mkdir coco
cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip panoptic_annotations_trainval2017.zip

cd annotations
unzip panoptic_train2017.zip
unzip panoptic_val2017.zip
rm panoptic_train2017.zip
rm panoptic_val2017.zip

cd ../
rm train2017.zip
rm val2017.zip
rm panoptic_annotations_trainval2017.zip