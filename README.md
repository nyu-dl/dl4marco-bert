# Passage Re-ranking with BERT

## Introduction
**\*\*\*\*\* Most of the code in this repository was copied from the original 
[BERT repository](https://github.com/google-research/bert).**\*\*\*\*\* 

This repository contains the code to reproduce our entry to the [MSMARCO passage
ranking task](http://www.msmarco.org/leaders.aspx), which was placed first with
a large margin over the second place. It also contains the code to reproduce our 
result on the [TREC-CAR dataset](http://trec-car.cs.unh.edu/), which is ~22 MAP 
points higher than the best entry from 2017 and a well-tuned BM25.

MSMARCO Passage Re-Ranking Leaderboard (Jan 8th 2019) | Eval MRR@10  | Dev MRR@10
------------------------------------- | :------: | :------:
1st Place - BERT (this code)          | **35.87** | **36.53**
2nd Place - IRNet                     | 28.06     | 27.80
3rd Place - Conv-KNRM                 | 27.12     | 29.02

TREC-CAR Test Set (Automatic Annotations) | MAP
----------------------------------------------------- | :------:
BERT (this code)                                      | **33.5**
BM25 [Anserini](https://github.com/castorini/Anserini/blob/master/docs/experiments-car17.md) | 15.6
[MacAvaney et al., 2017](https://trec.nist.gov/pubs/trec26/papers/MPIID5-CAR.pdf) (TREC-CAR 2017 Best Entry) | 14.8

The paper describing our implementation is [here](https://arxiv.org/abs/1901.04085).

## MS MARCO

### Download and extract the data
First, we need to download and extract MS MARCO and BERT files:
```
DATA_DIR=./data
mkdir ${DATA_DIR}

wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv -P ${DATA_DIR}
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P ${DATA_DIR}

tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.dev.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.eval.tar.gz -C ${DATA_DIR}
unzip ${DATA_DIR}/uncased_L-24_H-1024_A-16.zip -d ${DATA_DIR}
```

### Convert MS MARCO to TFRecord format
Next, we need to convert MS MARCO train, dev, and eval files to TFRecord files, 
which will be later consumed by BERT.

```
mkdir ${DATA_DIR}/tfrecord
python convert_msmarco_to_tfrecord.py \
  --output_folder=${DATA_DIR}/tfrecord \
  --vocab_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
  --train_dataset_path=${DATA_DIR}/triples.train.small.tsv \
  --dev_dataset_path=${DATA_DIR}/top1000.dev.tsv \
  --eval_dataset_path=${DATA_DIR}/top1000.eval.tsv \
  --dev_qrels_path=${DATA_DIR}/qrels.dev.tsv \
  --max_query_length=64\
  --max_seq_length=512 \
  --num_eval_docs=1000
```

This conversion takes 30-40 hours. Alternatively, you may download the
[TFRecord files here](https://drive.google.com/open?id=1IHFMLOMf2WqeQ0TuZx_j3_sf1Z0fc2-6) (~23GB).

### Training
We can now start training. We highly recommend using the free TPUs in
[our Google's Colab](https://drive.google.com/open?id=1vaON2QlidC0rwZ8JFrdciWW68PYKb9Iu).
Otherwise, a modern V100 GPU with 16GB cannot fit even a small batch size of 2
when training a BERT Large model.

In case you opt for not using the Colab, here is the command line to start 
training:
```
python run_msmarco.py \
  --data_dir=${DATA_DIR}/tfrecord \
  --bert_config_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --output_dir=${DATA_DIR}/output \
  --msmarco_output=True \
  --do_train=True \
  --do_eval=True \
  --num_train_steps=400000 \
  --num_warmup_steps=40000 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=1e-6
```

Training for 400k iterations takes approximately 70 hours on a TPU v2.
Alternatively, you can [download the trained model used in our submission here](https://drive.google.com/open?id=1crlASTMlsihALlkabAQP6JTYIZwC1Wm8) (~3.4GB).

You can also [download a BERT Base model trained on MS MARCO here](https://drive.google.com/open?id=1cyUrhs7JaCJTTu-DjFUqP6Bs4f8a6JTX). This model leads to ~2 points lower MRR@10 (34.7), but it is faster to train and evaluate. It can also fit on a single 12GB GPU.


## TREC-CAR

We describe in the next sections how to reproduce our results on the [TREC-CAR](http://trec-car.cs.unh.edu/) dataset.

### Downloading qrels, run and TFRecord files

The next steps (Indexing, Retrieval, and TFRecord conversion) take many hours.
Alternatively, you can skip them and download 
[the necessary files for training and evaluation here](https://drive.google.com/open?id=16tk7HmLaqvU0oIO5L_H8elwqKn2cJUzG) (~4.0GB), namely:
- queries (*.topics);
- query-relevant passage pairs (*.qrels);
- query-candidate passage pairs (*.run).
- TFRecord files (*.tf)

After downloading, you need to extract them to the TRECCAR_DIR folder:
```
TRECCAR_DIR=./treccar/
tar -xf treccar_files.tar.xz --directory ${TRECCAR_DIR}
```

And you are ready to go to the training/evaluation section.


### Downloading and Extracting the data

If you decided to index, retrieve and convert to the TFRecord format, you first
need to download and extract the TREC-CAR data:
```
TRECCAR_DIR=./treccar/
DATA_DIR=./data
mkdir ${DATA_DIR}

wget http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz -P ${TRECCAR_DIR}
wget http://trec-car.cs.unh.edu/datareleases/v2.0/train.v2.0.tar.xz -P ${TRECCAR_DIR}
wget http://trec-car.cs.unh.edu/datareleases/v2.0/benchmarkY1-test.v2.0.tar.xz -P ${TRECCAR_DIR}
wget https://storage.googleapis.com/bert_treccar_data/pretrained_models/BERT_Large_pretrained_on_TREC_CAR_training_set_1M_iterations.tar.gz -P ${DATA_DIR}


tar -xf  ${TRECCAR_DIR}/paragraphCorpus.v2.0.tar.xz
tar -xf  ${TRECCAR_DIR}/train.v2.0.tar.xz
tar -xf  ${TRECCAR_DIR}/benchmarkY1-test.v2.0.tar.xz
tar -xzf ${DATA_DIR}/BERT_Large_pretrained_on_TREC_CAR_training_set_1M_iterations.tar.gz
```

### Indexing TREC-CAR 

We need to index the corpus and retrieve documents using the BM25 algorithm for
each query so we have query-document pairs for training.

We index the TREC-CAR corpus using [Anserini](https://github.com/castorini/Anserini), 
an excelent toolkit for information retrieval research.

First, we need to install Maven, and clone and compile Anserini's repository:
```
sudo apt-get install maven
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
tar xvfz eval/trec_eval.9.0.4.tar.gz -C eval/ && cd eval/trec_eval.9.0.4 && make
cd ../ndeval && make
```

Now we can index the corpus (.cbor files):
```
sh target/appassembler/bin/IndexCollection -collection CarCollection \
-generator LuceneDocumentGenerator -threads 40 -input ${TRECCAR_DIR}/paragraphCorpus.v2.0 -index \
${TRECCAR_DIR}/lucene-index.car17.pos+docvectors+rawdocs -storePositions -storeDocvectors \
-storeRawDocs
```

You should see a message like this after it finishes:
```
2019-01-15 20:26:28,742 INFO  [main] index.IndexCollection (IndexCollection.java:578) - Total 29,794,689 documents indexed in 03:20:35
```

### Retrieving pairs of query-candidate document
We now retrieve candidate documents for each query using the BM25 algorithm.
But first, we need to convert the TREC-CAR files to a format that Anserini can 
consume.

First, we merge qrels folds 0, 1, 2, and 3 into a single file for training. 
Fold 4 will be the dev set.
```
for f in ${TRECCAR_DIR}/train/fold-[0-3]-base.train.cbor-hierarchical.qrels; do (cat "${f}"; echo); done >${TRECCAR_DIR}/train.qrels
cp ${TRECCAR_DIR}/train/fold-4-base.train.cbor-hierarchical.qrels ${TRECCAR_DIR}/dev.qrels
cp ${TRECCAR_DIR}/benchmarkY1/benchmarkY1-test/test.pages.cbor-hierarchical.qrels ${TRECCAR_DIR}/test.qrels
```

We need to extract the queries (first column in the space-separated files):
```
cat ${TRECCAR_DIR}/train.qrels | cut -d' ' -f1 > ${TRECCAR_DIR}/train.topics
cat ${TRECCAR_DIR}/dev.qrels | cut -d' ' -f1 > ${TRECCAR_DIR}/dev.topics
cat ${TRECCAR_DIR}/test.qrels | cut -d' ' -f1 > ${TRECCAR_DIR}/test.topics
```

And remove all duplicated queries:
```
sort -u -o ${TRECCAR_DIR}/train.topics ${TRECCAR_DIR}/train.topics
sort -u -o ${TRECCAR_DIR}/dev.topics ${TRECCAR_DIR}/dev.topics
sort -u -o ${TRECCAR_DIR}/test.topics ${TRECCAR_DIR}/test.topics
```

We now retrieve the top-10 documents per query for training and development sets.
```
nohup target/appassembler/bin/SearchCollection -topicreader Car -index ${TRECCAR_DIR}/lucene-index.car17.pos+docvectors+rawdocs -topics ${TRECCAR_DIR}/train.topics -output ${TRECCAR_DIR}/train.run -hits 10 -bm25 &

nohup target/appassembler/bin/SearchCollection -topicreader Car -index ${TRECCAR_DIR}/lucene-index.car17.pos+docvectors+rawdocs -topics ${TRECCAR_DIR}/dev.topics -output ${TRECCAR_DIR}/dev.run -hits 10 -bm25 &
```

And we retrieve top-1,000 documents per query for the test set.
```
nohup target/appassembler/bin/SearchCollection -topicreader Car -index ${TRECCAR_DIR}/lucene-index.car17.pos+docvectors+rawdocs -topics ${TRECCAR_DIR}/test.topics -output ${TRECCAR_DIR}/test.run -hits 1000 -bm25 &
```

After it finishes, you should see an output message like this:
```
(SearchCollection.java:166) - [Finished] Ranking with similarity: BM25(k1=0.9,b=0.4)
2019-01-16 23:40:56,538 INFO  [pool-2-thread-1] search.SearchCollection$SearcherThread (SearchCollection.java:167) - Run 2254 topics searched in 01:53:32
2019-01-16 23:40:56,922 INFO  [main] search.SearchCollection (SearchCollection.java:499) - Total run time: 01:53:36
```

This retrieval step takes 40-80 hours for the training set. We can speed it up
by increasing the number of threads (ex: -threads 6) and loading the index into
memory (-inmem option).

### Measuring BM25 Performance (optional)
To be sure that indexing and retrieval worked fine, we can measure the 
performance of this list of documents retrieved with BM25:
```
eval/trec_eval.9.0.4/trec_eval -m map -m recip_rank -c ${TRECCAR_DIR}/test.qrels ${TRECCAR_DIR}/test.run
```

It is important to use the -c option as it assigns a score of zero to queries
that had no passage returned.
The output should be like this:
```
map                   	all	0.1528
recip_rank            	all	0.2294
```

### Converting TREC-CAR to TFRecord

We can now convert qrels (query-relevant document pairs), run (
query-candidate document pairs), and the corpus into training, dev, and test 
TFRecord files that will be consumed by BERT.
(we need to install CBOR package: pip install cbor)
```
python convert_treccar_to_tfrecord.py \
  --output_folder=${TRECCAR_DIR}/tfrecord \
  --vocab_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/vocab.txt \
  --corpus=${TRECCAR_DIR}/paragraphCorpus/dedup.articles-paragraphs.cbor \
  --qrels_train=${TRECCAR_DIR}/train.qrels \
  --qrels_dev=${TRECCAR_DIR}/dev.qrels \
  --qrels_test=${TRECCAR_DIR}/test.qrels \
  --run_train=${TRECCAR_DIR}/train.run \
  --run_dev=${TRECCAR_DIR}/dev.run \
  --run_test=${TRECCAR_DIR}/test.run \
  --max_query_length=64\
  --max_seq_length=512 \
  --num_train_docs=10 \
  --num_dev_docs=10 \
  --num_test_docs=1000
```

This step requires at least 64GB of RAM as we load the entire corpus onto memory.


### Training/Evaluating

Before start training, you need to download a [BERT Large model pretrained on the training set of TREC-CAR](https://drive.google.com/open?id=1Ovc8DPtgQ411bUo-_UDSDVqpPsoWXvmG). This pretraining was necessary because the [official pre-trained BERT models](https://github.com/google-research/bert) were pre-trained on the full Wikipedia, and therefore they have seen, although in an unsupervised way, Wikipedia documents that are used in the test set of TREC-CAR. Thus, to avoid this leak of test data into training, we pre-trained the BERT re-ranker only on the half of Wikipedia used by TREC-CAR’s training set.

Similar to MS MARCO training, we made available [this Google Colab](https://colab.research.google.com/drive/1uIXKkxkEbwe2Z6-tGmbbH10ptwd2Tr0u) to train and evaluate on TREC-CAR. 

In case you opt for not using the Colab, here is the command line to start 
training:
```
python run_treccar.py \
  --data_dir=${TRECCAR_DIR}/tfrecord \
  --bert_config_file=${DATA_DIR}/uncased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=${DATA_DIR}/pretrained_models_exp898_model.ckpt-1000000 \
  --output_dir=${TRECCAR_DIR}/output \
  --trec_output=True \
  --do_train=True \
  --do_eval=True \
  --trec_output=True \
  --num_train_steps=400000 \
  --num_warmup_steps=40000 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=1e-6 \
  --max_dev_examples=3000 \
  --num_dev_docs=10 \
  --max_test_examples=None \
  --num_test_docs=1000
```

Because trec_output is set to True, this script will produce a
TREC-formatted run file "bert_predictions_test.run". We can evaluate the 
final performance of our BERT model using the official TREC eval tool, which 
is included in Anserini:
```
eval/trec_eval.9.0.4/trec_eval -m map -m recip_rank -c ${TRECCAR_DIR}/test.qrels ${TRECCAR_DIR}/output/bert_predictions_test.run
```

And the output should be:
```
map                   	all	0.3356
recip_rank            	all	0.4787
```

We made available [our run file here](https://drive.google.com/file/d/1bhTjtz_IK0ER5S-eV0AxyhjHCupLiukN/view?usp=sharing).

### Trained models
You can download our [BERT Large trained on TREC-CAR here](https://drive.google.com/open?id=1fzcL2nzUJMUd0w4J5JIeASSrN4uHlSqP).


#### How do I cite this work?
```
@article{nogueira2019passage,
  title={Passage Re-ranking with BERT},
  author={Nogueira, Rodrigo and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1901.04085},
  year={2019}
}
```
