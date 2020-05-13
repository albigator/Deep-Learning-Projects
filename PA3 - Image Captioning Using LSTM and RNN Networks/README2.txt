CSE 253 PA4

Description of files:

Notebooks:

starter_LSTM.ipynb - notebook for training baseline and stochastic LSTM
                   - also trains RNN (change import statement for AnnoNet.py)

Net_BLEU.ipynb - notebook for storing captions in json file and calculating BLEU scores
               - change paths for model files to run for LSTM or vanilla RNN
               - set prediction argument to 'd' for deterministic or 's' for stochastic captioning
               - last cell plots loss curves from training and validation
               
Net_Eval.ipynb - run evaluation on a single test image to get caption for report

Net_Test_Loss.ipynb - run evaluation on all test images to compute loss over entire set

Starter_GLOVE_FINAL.ipynb - notebook for training pre-trained embedding weights
               

Model Classes:

AnnoNet.py - class file for LSTM model

AnnoNetRNN.py - class file for RNN model

AnnoNet_GLOVE.py - class file for pre-trained GloVe embedding weights for LSTM


Dependencies:

data_loader.py - modified version of given dataloader file to load data and get ids

make_ids.py - make ids compatible with our dataloader

vocabulary_struct.py - file for vocab object and constructor

Vocab_File - file that holds generated vocab object

Note: These dependencies are for all notebooks except Starter_GLOVE_FINAL.ipynb.
      - This notebook requires separate dependencies provided in the link at the bottom
      

Pre-trained models, generated captions, and additional dependencies are provided at:

https://drive.google.com/open?id=1yTsaNufr0yJb0B3mOK7PmqL8jlKDWl71

