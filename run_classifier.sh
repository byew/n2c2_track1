#!/bin/bash

#Example script for running run_classifier.py


#for EPOCHS in 3 4 5 ; do
#	for LR in 2e-5 3e-5 5e-5; do
#		for BATCH_SZ in 16 32 ; do
#			MAX_SEQ_LEN=150
#
#			DATA_DIR=PATH/TO/MEDNLI/DATA/ #Modify this to be the path to the MedNLI data
#			OUTPUT_DIR=PATH/TO/OUTPUT/DIR/ #Modify this to be the path to your output directory
#			CLINICAL_BERT_LOC=PATH/TO/CLINICAL/BERT/MODEL #Modify this to be the path to the clinical BERT model
#
#			echo $OUTPUT_DIR
#
#			BERT_MODEL=clinical_bert # You can change this to biobert or bert-base-cased
#
#			mkdir -p $OUTPUT_DIR
#
#		  	python run_classifier.py \
#			  --data_dir=$DATA_DIR \
#			  --bert_model=$BERT_MODEL \
#			  --model_loc $CLINICAL_BERT_LOC \
#			  --task_name mednli \
#			  --do_train \
#			  --do_eval \
#			  --do_test \
#			  --output_dir=$OUTPUT_DIR  \
#			  --num_train_epochs $EPOCHS \
#			  --learning_rate $LR \
#			  --train_batch_size $BATCH_SZ \
#			  --max_seq_length $MAX_SEQ_LEN \
#			  --gradient_accumulation_steps 2
#		done
#	done
#done

EPOCHS=1
LR=2e-5
BATCH_SZ=32
MAX_SEQ_LEN=150

DATA_DIR=./data/
CLINICAL_BERT_LOC=./models/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/

BERT_MODEL=clinical_bert # You can change this to biobert or bert-base-cased


for FOLD_ID in 0 1 2 3 4 ; do
    OUTPUT_DIR="./models/cls.0628/$FOLD_ID"
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
      --fold_id $FOLD_ID
done