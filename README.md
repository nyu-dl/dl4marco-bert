# Passage Re-ranking with BERT

## Introduction
**\*\*\*\*\* Most of the code in this repository was copied from the original 
[BERT repository](https://github.com/google-research/bert).**\*\*\*\*\* 

This repository contains the code to reproduce our entry to the [MSMARCO passage
ranking task](http://www.msmarco.org/leaders.aspx), which was placed first with
a large margin over the second place.

MSMARCO Passage Re-Ranking Leaderboard (Jan 8th 2019) | Eval MRR@10  | Eval MRR@10
------------------------------------- | :------: | :------:
1st Place - BERT (this code)          | **35.87** | **36.53**
2nd Place - IRNet                     | 28.06     | 27.80
3rd Place - Conv-KNRM                 | 27.12     | 29.02

The paper describing our implementation is [here](https://arxiv.org/abs/1901.04085).

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

## Convert MS MARCO to tfrecord format
Next, we need to convert MS MARCO train, dev, and eval files to tfrecord files, 
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
[tfrecord files here](https://drive.google.com/open?id=1IHFMLOMf2WqeQ0TuZx_j3_sf1Z0fc2-6) (~23GB).

## Training
We can now start training. We highly recommend using TPUs, which are free in
[Google's Colab](https://drive.google.com/open?id=1vaON2QlidC0rwZ8JFrdciWW68PYKb9Iu).
Otherwise, a modern V100 GPU with 16GB cannot fit even a small batch size of 2
when training a BERT Large model.

In case you opt for not using the Colab, here is the command line to start 
training:
```
python run.py \
  --data_dir=${DATA_DIR}/tfrecord \
  --bert_config_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --output_dir=${DATA_DIR}/output \
  --msmarco_output=True \
  --do_train=True \
  --do_eval=True \
  --num_train_steps=400000 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=1e-6
```

Training for 400k iterations takes approximately 70 hours on a TPU v2.
Alternatively, you can [download the trained model used in our submission here](https://drive.google.com/open?id=1crlASTMlsihALlkabAQP6JTYIZwC1Wm8) (~3.4GB).

## TREC-CAR
You can download our [BERT Large trained on TREC-CAR here](https://drive.google.com/open?id=1fzcL2nzUJMUd0w4J5JIeASSrN4uHlSqP).
We also made available a [BERT Large model pretrained on the training set of TREC-CAR](https://drive.google.com/open?id=1Ovc8DPtgQ411bUo-_UDSDVqpPsoWXvmG). This pretraining was necessary because the [official pre-trained BERT models](https://github.com/google-research/bert) were pre-trained on the full Wikipedia, and therefore they have seen, although in an unsupervised way, Wikipedia documents that are used in the test set of TREC-CAR. Thus, to avoid this leak of test data into training, we pre-trained the BERT re-ranker only on the half of Wikipedia used by TREC-CAR’s training set.

#### How do I cite this work?
```
@article{nogueira2019passage,
  title={Passage Re-ranking with BERT},
  author={Nogueira, Rodrigo and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1901.04085},
  year={2019}
}
```
