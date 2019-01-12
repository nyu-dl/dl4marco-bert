from collections import OrderedDict


######################
# General parameters #
######################
data_folder = '/scratch/rfn216/NeuralResearcher/data/'
metrics_map = OrderedDict([('MAP', 0), ('RPrec', 1), ('MRR', 2), ('NDCG', 3), ('MRR@10', 4)])


#################################
# TF-Record Conversion for BERT #
#################################
tfrecord_folder = data_folder + '/bert/tfrecord_3868/'
bert_vocab_file = data_folder + '/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt'
max_training_docs = 10  # Number of maximum docs per query for training.
max_eval_docs = 1000  # Number of maximum docs per query for eval.
max_length_query = 64
max_length_doc = 512 - max_length_query


#############################################
# MS-MARCO to TF-Record Conversion for BERT #
#############################################
train_dataset_path = data_folder + '/msmarco/triples.train.small.tsv' # Path to load training dataset containing the tab separated <query, positive_paragraph, negative_paragraph> tuples.
valid_dataset_path = data_folder + '/msmarco/top1000.dev.tsv' # Path to valid dataset with ~6600 queries and top-1000 documents per query, out of order.
valid_qrels_path = data_folder + '/msmarco/qrels.dev.tsv'  # Path to the query_id relevant doc ids mapping.
test_dataset_path = data_folder + '/msmarco/top1000.eval.tsv'  # Path to test dataset with top-1000 documents per query, out of order.
msmarco_corpus_txt = data_folder + '/msmarco_train_corpus_3869.txt'
