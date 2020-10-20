
#basics
import pandas as pd
import torch 


def extract_features(data:pd.DataFrame, max_sample_length:int, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    
    
    train = data[data["split"]=="train"]
    test = data[data["split"]=="test"]
    val = data[data["split"]=="val"]
    
    list1="abcdefghijklmnopqrstuvlwmnohyzx*:[]'`~!;.,#+1234567890+-_()¨<>/&%)= "
    character_ids= {char:idx for idx,char in enumerate(list1)}
    word_max_length=max([len(id2word[ids]) for ids in data["token_id"].tolist()])
   

    def samples(data):

        n = data["sentence_id"].unique()
        samples = []
        for i in n: 
            sent = data.loc[data["sentence_id"].isin([i])]
            word_ids = sent["token_id"].tolist()
            sent_encode = []
            for i in word_ids:
                word = id2word[i]
                encode=[]
                
                for char in word:
                    if char.isalpha():
                        char = char.lower()
                    else: 
                        char = str(char)
                    char_id = character_ids[char] 
                    encode.append(char_id)
      
               
                tensor_en = torch.LongTensor(encode)
                a = torch.cat([tensor_en, tensor_en.new_zeros(word_max_length - tensor_en.size(0),dtype= torch.long)],0) # pad the tensor for word to match max length of word
                sent_encode.append(a)
                

            b = torch.stack(sent_encode,0)             
            p=torch.cat([b,torch.zeros(max_sample_length - len(sent_encode),word_max_length,dtype= torch.long)],0) # pad tensors for sentences to match max length of sentence
          
   
            samples.append(p)
        final= torch.stack(samples)
        return final 
    
    device = torch.device('cuda:1')
    
    train_x = samples(train)
    train_x= train_x.to(device)
    
    test_x = samples(test)
    test_x= test_x.to(device)
    
    val_x = samples(val)
    val_x = val_x.to(device)
    return train_x, val_x, test_x




