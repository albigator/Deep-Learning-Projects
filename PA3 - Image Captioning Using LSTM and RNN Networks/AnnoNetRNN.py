import torch
from torch import LongTensor
from torch.nn import Embedding, LSTM
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models

class AnnoNet(nn.Module):

    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, hidden_units=1, feature_extract=True):
        super().__init__()
        
        self.batch_size = batch_size
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        #self.embed = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.embed = nn.Linear(in_features = resnet.fc.in_features, out_features = embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=hidden_units, batch_first = True)
        self.vocab_decoder = nn.Linear(in_features = hidden_dim, out_features = vocab_size)
        
        
    #make embedding dim 60 (57 was max train caption length)
    def forward(self, x, captions, lengths):
        with torch.no_grad():
            features = self.resnet(x)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.embed(features)) 
        #features = self.embed(features)
        
        embed = self.word_embeddings(captions)
        embed = torch.cat((features.unsqueeze(1), embed), 1)
        embedded_sequence = pack_padded_sequence(embed, lengths, batch_first=True)
        lstm_outputs, _ = self.rnn(embedded_sequence)
        output = self.vocab_decoder(lstm_outputs.data)
        return output
    
    def eval_pass(self, x, states=None, prediction='d', temp=0.2):
        with torch.no_grad():
            features = self.resnet(x)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.embed(features)) 
        #features = self.embed(features)
        
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(60):
            hiddens, states = self.rnn(inputs, states)         
            outputs = self.vocab_decoder(hiddens.squeeze(1)) 
            #print(outputs.shape)
            
            # Deterministic
            if prediction == 'd':
                _, predicted = outputs.max(1) 
            
            # Stochastic
            if prediction == 's':
                softmaxed_output = F.softmax(outputs/temp, dim = 1)
                #print(softmaxed_output.shape)
                p_dist = torch.distributions.categorical.Categorical(probs=softmaxed_output)
                predicted = p_dist.sample()  
            
            #print("Stochastic Prediction: ", predicted)
            #print("Deterministic Prediction: ", predicted)                        
            sampled_ids.append(predicted)
            inputs = self.word_embeddings(predicted)                      
            inputs = inputs.unsqueeze(1) 
            if predicted == 25120:
                break
        sampled_ids = torch.stack(sampled_ids, 1)               
        return sampled_ids