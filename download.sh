#!/bin/bash

mkdir -p data
wget -c "https://zenodo.org/record/5140038/files/avt_image_db.tar.gz?download=1" -O data/avt_image_db.tar.gz

wget -c "https://zenodo.org/record/5140038/files/image_simplicity.tar.gz?download=1" -O data/image_simplicity.tar.gz

wget -c "https://zenodo.org/record/5140038/files/rule_of_thirds.tar.gz?download=1" -O data/rule_of_thirds.tar.gz

wget -c "https://zenodo.org/record/5140038/files/models.tar.lzma?download=1" -O models.tar.lzma


cd data

tar -xf avt_image_db.tar.gz
tar -xf image_simplicity.tar.gz
tar -xf rule_of_thirds.tar.gz

cd ..


tar -xvf models.tar.lzma
