#!/bin/bash


EPOCHS=5
LR=2e-5
BATCH_SZ=32
MAX_SEQ_LEN=150

DATA_DIR=./data/
CLINICAL_BERT_LOC=./models/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/

OUTPUT_DIR_PREFIX=$1
#DEVICE=$2

BERT_MODEL=clinical_bert # You can change this to biobert or bert-base-cased


#for FOLD_ID in 0 1 2 3 4 ; do
#    OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
#    echo $OUTPUT_DIR
#    mkdir -p $OUTPUT_DIR
#
#
#    python main.py \
#      --data_dir=$DATA_DIR \
#      --bert_model=$BERT_MODEL \
#      --model_loc $CLINICAL_BERT_LOC \
#      --task_name n2c2 \
#      --do_train \
#      --do_eval \
#      --output_dir=$OUTPUT_DIR  \
#      --num_train_epochs $EPOCHS \
#      --learning_rate $LR \
#      --train_batch_size $BATCH_SZ \
#      --max_seq_length $MAX_SEQ_LEN \
#      --fold_id $FOLD_ID \
#      --device $DEVICE
#done

(
FOLD_ID=0
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


python main.py \
  --data_dir=$DATA_DIR \
  --bert_model=$BERT_MODEL \
  --model_loc $CLINICAL_BERT_LOC \
  --task_name n2c2 \
  --do_train \
  --do_eval \
  --output_dir=$OUTPUT_DIR  \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --train_batch_size $BATCH_SZ \
  --max_seq_length $MAX_SEQ_LEN \
  --fold_id $FOLD_ID \
  --device cuda:1
) &
(
FOLD_ID=1
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


python main.py \
  --data_dir=$DATA_DIR \
  --bert_model=$BERT_MODEL \
  --model_loc $CLINICAL_BERT_LOC \
  --task_name n2c2 \
  --do_train \
  --do_eval \
  --output_dir=$OUTPUT_DIR  \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --train_batch_size $BATCH_SZ \
  --max_seq_length $MAX_SEQ_LEN \
  --fold_id $FOLD_ID \
  --device cuda:2
) &
(
FOLD_ID=2
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


python main.py \
  --data_dir=$DATA_DIR \
  --bert_model=$BERT_MODEL \
  --model_loc $CLINICAL_BERT_LOC \
  --task_name n2c2 \
  --do_train \
  --do_eval \
  --output_dir=$OUTPUT_DIR  \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --train_batch_size $BATCH_SZ \
  --max_seq_length $MAX_SEQ_LEN \
  --fold_id $FOLD_ID \
  --device cuda:3
) &
(
FOLD_ID=3
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


python main.py \
  --data_dir=$DATA_DIR \
  --bert_model=$BERT_MODEL \
  --model_loc $CLINICAL_BERT_LOC \
  --task_name n2c2 \
  --do_train \
  --do_eval \
  --output_dir=$OUTPUT_DIR  \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --train_batch_size $BATCH_SZ \
  --max_seq_length $MAX_SEQ_LEN \
  --fold_id $FOLD_ID \
  --device cuda:4
) &
(
FOLD_ID=4
OUTPUT_DIR="$OUTPUT_DIR_PREFIX/$FOLD_ID"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR


python main.py \
  --data_dir=$DATA_DIR \
  --bert_model=$BERT_MODEL \
  --model_loc $CLINICAL_BERT_LOC \
  --task_name n2c2 \
  --do_train \
  --do_eval \
  --output_dir=$OUTPUT_DIR  \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --train_batch_size $BATCH_SZ \
  --max_seq_length $MAX_SEQ_LEN \
  --fold_id $FOLD_ID \
  --device cuda:5
)