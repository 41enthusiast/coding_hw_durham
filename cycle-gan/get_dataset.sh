URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
ZIP_FILE=./datasets/horse2zebra.zip
TARGET_DIR=./datasets/horse2zebra/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE