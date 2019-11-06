#!/bin/bash

NET_NAME='osnet'
PT_NET='../model/osnet.py'
PT_CKPT=''
INPUT_SHAPE='1,3,244,244'

SAVE_PATH='.'
CAFFE_PROTOTXT=${SAVE_PATH}/${NET_NAME}.prototxt
CAFFE_MODEL=${SAVE_PATH}/${NET_NAME}.caffemodel

python ../pt2caffe/pytorch_to_caffe.py \
    --net-name ${NET_NAME} \
    --pt-net ${PT_NET} \
    --pt-ckpt ${PT_CKPT} \
    --input-shape ${INPUT_SHAPE} \
    --caffe-prototxt ${CAFFE_PROTOTXT} \
    --caffe-model ${CAFFE_MODEL}

sed -i 's/ceil_mode: false/round_mode: FLOOR/g' ${CAFFE_PROTOTXT}
