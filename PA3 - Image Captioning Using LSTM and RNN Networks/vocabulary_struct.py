import nltk
from pycocotools.coco import COCO

class vocabulary():
    '''Vocabulary class for storing word IDs'''
    def __init__(self):
        #create mapping dictionaries
        self.word2index = {}
        self.index2word = {}
        #length of indices
        self.num_inds = 0
    
    #update the dictionaries with a new word
    def update(self,word):
        assert(isinstance(word,str))
        assert(len(word)>0)
        if word not in self.word2index:
            self.word2index[word] = self.num_inds
            self.index2word[self.num_inds] = word
            self.num_inds += 1
    
    #call to return a word index given a word
    def __call__(self,word):
        assert(isinstance(word,(str, int)))
        if isinstance(word,str):
            assert(len(word)>0)
            if word not in self.word2index:
                return self.word2index['<unk>']
            else:
                return self.word2index[word]
        else:
            assert(word >= 0)
            try:
                return self.index2word[word]
            except:
                print(str(word)+' is not a valid index')
        
    def __len__(self):
        return self.num_inds
    
def create_vocab(json):
    coco = COCO(json)
    ids = coco.anns.keys()
    vocab = vocabulary()
    print('building vocabulary')
    maximum = 0
    for i, id in enumerate(ids):
        cap = str(coco.anns[id]['caption'])
        tokens = nltk.word_tokenize(cap.lower())
        temp = len(tokens)
        if maximum < temp:
            maximum = temp
            print(i)
        for word in tokens:
            vocab.update(word)
    vocab.update('<unk>')
    vocab.update('<end>')
    vocab.update('<start>')
    vocab.update('<pad>')
    print(maximum)
    return vocab
    