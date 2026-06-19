#!/usr/bin/env bash
# Download KITTI Object Detection (the small "raw"/data_object part is ~12 GB).
# NOTE: KITTI now requires accepting a license on the download page. If the direct
# URLs below 403, register at https://www.cvlibs.net/datasets/kitti/ and download
# "Download Object Detection dataset" manually into data/kitti with this layout:
#   data/kitti/{image_2,velodyne,label_2,calib}/
set -e
DEST=${1:-data/kitti}
mkdir -p "$DEST"
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti

echo "Downloading KITTI Object Detection into $DEST (~12 GB)..."
echo "If downloads fail with 403, see the note at the top of this script."

wget -c "$BASE/data_object_image_2.zip"        -O "$DEST/image_2.zip"
wget -c "$BASE/data_object_velodyne.zip"       -O "$DEST/velodyne.zip"
wget -c "$BASE/data_object_label_2.zip"       -O "$DEST/label_2.zip"
wget -c "$BASE/data_object_calib.zip"         -O "$DEST/calib.zip"

echo "Unzipping..."
unzip -o -q "$DEST/image_2.zip"   -d "$DEST"
unzip -o -q "$DEST/velodyne.zip"  -d "$DEST"
unzip -o -q "$DEST/label_2.zip"   -d "$DEST"
unzip -o -q "$DEST/calib.zip"     -d "$DEST"

# Flatten into the layout paired_loader.py expects (training split):
mkdir -p "$DEST/image_2" "$DEST/velodyne" "$DEST/label_2" "$DEST/calib"
[ -d "$DEST/training/image_2" ]  && cp "$DEST"/training/image_2/*.png   "$DEST/image_2/"  2>/dev/null || true
[ -d "$DEST/training/velodyne" ] && cp "$DEST"/training/velodyne/*.bin   "$DEST/velodyne/" 2>/dev/null || true
[ -d "$DEST/training/label_2" ] && cp "$DEST"/training/label_2/*.txt    "$DEST/label_2/"   2>/dev/null || true
[ -d "$DEST/training/calib" ]    && cp "$DEST"/training/calib/*.txt      "$DEST/calib/"     2>/dev/null || true

echo "Done. Verify: ls $DEST"
echo "Next: python -m train.trainer --config config/early_2d.yaml"