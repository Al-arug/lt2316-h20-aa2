
#basics

import pandas as pd
import torch
from lxml import etree as ET
from glob import glob
from nltk import word_tokenize
from string import punctuation
from nltk.tokenize import WhitespaceTokenizer
import string
from venn import venn
import random
import matplotlib.pyplot as plt


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        #self.get_y()
        #self.plot_split_ner_distribution()
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)
      
        
    def My_tokenizer(self,sentence,charoff=False):     #  I removed an erlier version of this tokenizer that contains a set of rules that could capture name entities. Now it's just an nltk tokenizer 
        
        numbers = [str(i) for i in range(1000)]
        tokens=[]
        char_o = []
        tokenized= word_tokenize(sentence)
        
        for i in tokenized:
            if i[-1] in string.punctuation:
                tokens.append(i[:-1])
            else:
                tokens.append(i)
                
        

        c_tokens=[ token for token in tokens if not token in string.punctuation if not token in numbers]
        l_tokens= " ".join([t + " " for t in c_tokens])
        b = WhitespaceTokenizer().tokenize(l_tokens)
        if charoff:  
            if len(c_tokens)>1:
                m = list(WhitespaceTokenizer().span_tokenize(l_tokens))
                char_o = list(zip(b,m))
            return char_o
        else: 
            return b

    
    def medVocab_and_split(self):                               # This function splits the data and creat medical vocab corpus 
        train1 = glob("{}/Train/*/*.xml".format(self.data_dir)) 
        
        val_size = (len(train1)*20)//100
        self.val = random.sample(train1,k= val_size)
        self.train = [i for i in train1 if i not in self.val]
        
        self.test = glob("{}/Test/*/*/*.xml".format(self.data_dir))
        
        allfiles = train1 + self.test
        self.med_vocab={}
        sentences=[]
        for file in allfiles:   
            root = ET.parse(file).getroot() 
            for child in root.findall("sentence"):
                text = child.items()[1][1] 
                sentences.append(text)
                for i in child.findall("entity"):
                    entity_type = i.items()[2][1]   
                    self.med_vocab[i.items()[3][1]] = entity_type 
                    
        voc=[]                # this part collects all words in all splits to make ids 
        for sentence in sentences:
            voc+= self.My_tokenizer(sentence,charoff=False)        
        self.word2id={token:idx for idx,token in enumerate(set(voc))}
       # self.vocab=[token for token in self.word2id.keys()]
        self.ner2id={"N":1,"drug_n":2,"drug":3,"group":4,"brand":5}
    
        pass



    def _parse_data(self,data_dir):
        self.data_dir = data_dir  
        
        def sort(self,files,split):            
            data_df={"sentence_id":[],"token_id":[],"char_start_id":[],"char_end_id":[],"split":[]} 
            ner_df={"sentence_id":[],"ner_id":[],"char_start_id":[],"char_end_id":[]}
            repeated_ids =[]
            for file in files:   
                    root = ET.parse(file).getroot() 
                    for child in root.findall("sentence"):
                        text = child.items()[1][1]
                        sent_id= child.items()[0][1] 
                        if len(text)>3 and sent_id not in repeated_ids: #Avoid repeated sentence ids in a split. 
                            repeated_ids.append(sent_id) 
                            tokens_char = self.My_tokenizer(text,charoff=True)
                           
                            for i in tokens_char:
                                data_df['token_id'].append(self.word2id[i[0]])
                                data_df['char_start_id'].append(i[1][0])
                                data_df['char_end_id'].append(i[1][1])
                                data_df['sentence_id'].append(sent_id)
                                ner_df['char_start_id'].append(i[1][0])
                                ner_df['char_end_id'].append(i[1][1])
                                ner_df['sentence_id'].append(sent_id)
                                data_df["split"].append(split)
                                if i[0] in self.med_vocab: 
                                    Ner = self.ner2id[self.med_vocab[i[0]]]
                                else:
                                    Ner = self.ner2id["N"]
                        
                                ner_df["ner_id"].append(Ner)
                            
            ner_labels = pd.DataFrame.from_dict(ner_df)                
            data = pd.DataFrame.from_dict(data_df) 
            ner_only = ner_labels[ner_labels["ner_id"]!=1]
            
            return ner_only, ner_labels , data
        
        self.medVocab_and_split()
        self.ners_tst, self.labeled_tst, id_test = sort(self,self.test,"test")
        self.ners_tr, self.labeled_train , id_train = sort(self,self.train,"train")
        self.ners_val, self.labeled_val , id_val = sort(self,self.val,"val")
    
        

    
        self.ner_df = pd.concat([self.ners_tr,self.ners_tst,self.ners_val],axis=0)
        self.data_df = pd.concat([id_train,id_test,id_val],axis=0)
       # self.labeled_df = pd.concat([self.labeled_train,self.labeled_tst,self.labeled_val],axis=0)
        
        
        self.id2ner = {ids:ners for ners,ids in self.ner2id.items()}
        self.id2word = {ids:tokens for tokens,ids in self.word2id.items()}
        self.vocab = [self.id2word[i] for i in id_train["token_id"].tolist()]  # vocab of the train set only
        
        
        
        lengths=[]
        n = self.data_df["sentence_id"].unique()
        for i in n: 
            b = self.data_df.loc[self.data_df["sentence_id"].isin([i])]
            lengths.append(len(b["token_id"].tolist()))
            
        self.max_sample_length = max(lengths)  
        
        
        
        
        pass
   

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
 
        # i.new_zeros(max_len - i.size(0),dtype=torch.long)
        def pad(self,data,max_len):

            n = data["sentence_id"].unique()  #torch.LongTensor([-1]*(max_len - i.size(0)))
            sample_tags = []                  #torch.LongTensor([-1]*(max_len - i.size(0)))
            for i in n:
                b = data.loc[data["sentence_id"].isin([i])]
                sample_tags.append(torch.LongTensor(b["ner_id"].tolist())) 
            tags = torch.stack([torch.cat([i,i.new_zeros(max_len - i.size(0),dtype=torch.long)], 0) for i in sample_tags],0)
            return tags
            
        device = torch.device('cuda:1')
 
        self.train_y =  pad(self,self.labeled_train,self.max_sample_length) 
        self.train_y = self.train_y.to(device)
        self.test_y = pad(self,self.labeled_tst,self.max_sample_length) 
        self.test_y= self.test_y.to(device)
        self.val_y = pad(self,self.labeled_val,self.max_sample_length) 
        self.val_y = self.val_y.to(device)
        return self.train_y, self.val_y, self.test_y
             
        
    def plot_split_ner_distribution(self):
        
        self.get_y()
        
        def count(df):
            token= df.groupby("ner_id").count()
            a = ["n_drug","drug","group","brand"]
            b = token["sentence_id"].tolist()
            c= zip(a,b)
            final = {i[0]:i[1] for i in c}
 
            return final

        train = count(self.ners_tr)
        test = count(self.ners_tst)
        val = count(self.ners_val)
        
        plotting= pd.DataFrame([train,test,val],index=['train', 'test', 'val'])
        plotting.plot.bar(figsize=(4,8))
        plt.show()
        
        pass 


    def plot_sample_length_distribution(self):
        
        lengths=[]
        n = self.data_df["sentence_id"].unique()
        for i in n: 
            b = self.data_df.loc[self.data_df["sentence_id"].isin([i])]
            lengths.append(len(b["token_id"].tolist()))

        plt.hist(lengths,80)
        plt.show()
 
        pass


    def plot_ner_per_sample_distribution(self): 
        
        a = self.ner_df.groupby("sentence_id").count()
        b = a["ner_id"].tolist()
        
        for i in self.data_df["sentence_id"].unique():
            if i not in self.ner_df["sentence_id"].tolist():
                b.append(0)
                
        plt.hist(b,80)
        plt.show()
       
        pass


    def plot_ner_cooccurence_venndiagram(self):

        n_drug = self.ner_df.loc[self.ner_df['ner_id'] == 2, 'sentence_id'].tolist()
        drug = self.ner_df.loc[self.ner_df['ner_id'] == 3, 'sentence_id'].tolist()
        group = self.ner_df.loc[self.ner_df['ner_id'] == 4, 'sentence_id'].tolist()
        brand = self.ner_df.loc[self.ner_df['ner_id'] == 5, 'sentence_id'].tolist() 

        venn({"n_drug": set(n_drug), "drug": set(drug), "group": set(group), "brand": set(brand)})
        plt.show()
        
        pass



