# Kaomoji Detection

BiLSTM-CRF for Kaomoji Detection

# Files

* kaomoji_tagger.py  
  BiLSTM-CRF model
* train.py  
  training script
* make_iterator.py  
  use torchtext for batch processing.
* preprocess.sh  
  preprocessing shell script.
* pickle2tsv.py  
  convert pickle data to tsv data.
* filter.py  
  remove lines with different numbers of tags and chars.
* dataset_clean.txt  
  cleaned dataset (Line 29490, 40929 are deleted).
* dataset_random.txt  
  shuffled dataset.
* train.txt, valid.txt  
  splited dataset extracted from dataset_random.txt.

# Usage

    bash preprocess.sh    # make train.txt, valid.txt
    GPU_ID=1
    CUDA_VISIBLE_DEVICES=$GPUID python train.py -gpu 0

# Options

`python train.py -h`

    usage: train.py [-h] [-no_CRF] [-no_features] [-optimizer {SGD,Adam}]
    [-gpu GPU] [-epoch EPOCH] [-batch_size BATCH_SIZE]
    [-embed_size EMBED_SIZE] [-feat_embed_size FEAT_EMBED_SIZE]
    [-hidden_size HIDDEN_SIZE] [-train_file TRAIN_FILE]
    [-valid_file VALID_FILE] [-test_file TEST_FILE]
    [-out_dir OUT_DIR] [-resume RESUME] [-tensorboard]
    [-test_only]
    
    optional arguments:
    -h, --help            show this help message and exit
    -no_CRF               Flag CRF layer [default: use CRF]
    -no_features          Flag not use features [default: use features]
    -optimizer {SGD,Adam}
        Select Optimizer SGD or Adam [default: SGD]
    -gpu GPU              GPU ID [default: -1]
    -epoch EPOCH          Num of Epoch [default: 5]
    -batch_size BATCH_SIZE 
                          Batch size [default: 512]
    -embed_size EMBED_SIZE
                          Size of embedding dimension [default: 128]
    -feat_embed_size FEAT_EMBED_SIZE
                          Size of embedding dimension [default: 16]
    -hidden_size HIDDEN_SIZE
                          Number of node in hidden layers [default: 128]
    -train_file TRAIN_FILE
                          Train dataset file [default: data/train.txt]
    -valid_file VALID_FILE
                          Validation dataset file [default: data/valid.txt]
    -test_file TEST_FILE  Test dataset file [default: None]
    -out_dir OUT_DIR      Directory to Output [default: result]
    -resume RESUME        Resume trained model [default: None]
    -tensorboard          Logging to tensorboard (pip install tensorboardX) [default: False]
    -test_only

# Reference

[GitHub:BiLSTM-CRF on PyTorch](https://github.com/kaniblu/pytorch-bilstmcrf)
