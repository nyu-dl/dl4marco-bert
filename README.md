# BERT as Passage-Reranker

## Introduction
**\*\*\*\*\* Most of the code in this repository was copied from the original 
[BERT repository](https://github.com/google-research/bert).**\*\*\*\*\* 

This repository contains the code to reproduce our entry to the [MSMARCO passage
ranking task](http://www.msmarco.org/leaders.aspx)

The paper describing our implementation is [here]().


MSMARCO Passage Re-Ranking Leaderboard (Jan 8th 2019) | Eval MRR@10  | Eval MRR@10
------------------------------------- | :------: | :------:
1st Place - BERT (this code)          | **35.87** | **36.53**
2nd Place - IRNet                     | 28.06     | 27.80
3rd Place - Conv-KNRM                 | 27.12     | 29.02

## Download and extract the data
First, we need to download and extract MS MARCO and BERT files:
```
DATA_DIR=./data
mkdir ${DATA_DIR}

wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv -P ${DATA_DIR}
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P ${DATA_DIR}

tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.dev.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.eval.tar.gz -C ${DATA_DIR}
unzip ${DATA_DIR}/uncased_L-24_H-1024_A-16.zip -d ${DATA_DIR}
```

## Convert MS MARCO to tfrecord
Next, we need to convert MS MARCO train, dev, and eval file to tfrecord files, 
which will be later consumed by BERT.

```
mkdir ${DATA_DIR}/tfrecord
python convert_msmarco_to_tfrecord.py \
  --tfrecord_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
  --train_dataset_path=${DATA_DIR}/triples.train.small.tsv \
  --dev_dataset_path=${DATA_DIR}/top1000.dev.tsv \
  --eval_dataset_path=${DATA_DIR}/top1000.eval.tsv \
  --dev_qrels_path=${DATA_DIR}/qrels.dev.tsv \
  --max_query_length=64\
  --max_seq_length=512 \
  --num_eval_docs=1000
```

This conversion takes 30-40 hours. Alternatively, you can download the
[tfrecords file here]() (~23GB):

## Training
We can now start training. We highly recommend to use a TPU, which are free in
[Google's colab](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb). Otherwise, a modern V100 GPU with 16GB
cannot fit even a small batch size of 2 when training a BERT Large model.

```
TFRECORD_FOLDER=${DATA_DIR}/tfrecord
mkdir ${TFRECORD_FOLDER}
python run.py \
  --data_dir=${TFRECORD_FOLDER} \
  --vocab_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --output_dir=${DATA_DIR}/output \
  --msmarco_output=True \
  --do_train=True\
  --do_eval=True\
  --num_train_steps=400000\
  --
```

Training for 400k iterations takes approximately 70 hours on a TPU v2.
Alternatively, you can [download the trained model used in our submission here](https://storage.googleapis.com/bert_msmarco_data/pretrained_models/trained_bert_large.zip) (~3.4GB).

#### How do I cite this work?
```
@article{nogueira2019passage,
  title={Passage Re-ranking with BERT},
  author={Nogueira, Rodrigo and Cho, Kyunghyun},
  journal={arXiv preprint},
  year={2019}
}
```
