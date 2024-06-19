#!/bin/bash

PTH=../../../magface_iresnet18_casia_dp.pth
FEAT_SUFFIX=features
NL=18

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw/img.list \
                    --feat_list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${PTH}
