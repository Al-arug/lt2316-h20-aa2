
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
class Trainer:

    
    def __init__(self, dump_folder):
        self.dump_folder = dump_folder
        self.device = torch.device('cuda:1')
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))

    def load_model(self,path):
        # Finish this function so that it loads a model and return the appropriate variables
        self.path = path
        m = torch.load(path)
# - or -
      #  model.train()
        
        return m 


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        
        embedding_size = 60
        hidden_size = 51
       # epochs =  3    
        output_size = 6
        drop_out = hyperparamaters["drop_out"]
        number_layers = hyperparamaters["number_layers"]
        batch_size =  hyperparamaters["batch_size"]
        epochs =  hyperparamaters["epochs"]
        optim_type =  hyperparamaters["optim"] 
        lr = hyperparamaters["lr"] 
        
        
        train_y = torch.split(train_y, batch_size)
        train_X = torch.split(train_X, batch_size)    
            
            

        model = model_class(embedding_size, hidden_size,output_size, drop_out,number_layers )
        model = model.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(self.device)
        
        if optim_type == 1:
            optimizer = optim.Adam(model.parameters(),lr= lr)
        else: 
            optimizer = optim.SGD(model.parameters(), lr=lr , momentum=0.9)
       # optimizer = optim.Adam(model.parameters(),lr= 0.001)
       # prediction=[]
       # labels=[]
        
        for e in range(epochs):  
            total_loss=0
            model.train() 
            for idx,x in enumerate(train_X):
                optimizer.zero_grad()
                predictions = model(x.float())
                indecies = torch.argmax(predictions, dim = -1) 
                labels = train_y[idx]
                labels = labels.reshape(-1)
                loss = loss_function(predictions,labels)
                loss.backward()       
                optimizer.step()
                total_loss += loss.item()
        print(total_loss)
        val_y = torch.split(val_y, batch_size)
        val_X = torch.split(val_X, batch_size)

        pred=[]
        label= []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_X):
                predictions = model(batch.float())
                label+=val_y[i].tolist()
                indecies = torch.argmax(predictions, dim =1) 
         #print(indecies)
                pred+=indecies.tolist()
    
        labels = [i for l in label for i in l]
        
        a = accuracy_score(labels,pred)
        p = precision_score(labels,pred, average='weighted')
        f1= f1_score(labels,pred, average='weighted')
        r = recall_score(labels,pred, average='weighted')
        
        scores={"accuracy":a,"recall":r,"f1":f1,"percision":p  } 
        #,"loss":int(total_loss)
        #print(scores)
        self.save_model(epochs, model, optimizer, total_loss, scores, hyperparamaters, hyperparamaters["name"])
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        

        
        embedding_size = 60
        hidden_size = 51 
        output_size = 6
        
        
        parameters= self.load_model(best_model_path)
        hyperparamaters = parameters["hyperparamaters"]
        model_state_dict = parameters['model_state_dict']
        
        

        drop_out = hyperparamaters["drop_out"]
        number_layers = hyperparamaters["number_layers"]
        batch_size =  hyperparamaters["batch_size"]
        epochs =  hyperparamaters["epochs"]
        

        model = model_class(embedding_size, hidden_size,output_size, drop_out,number_layers )
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)
        
                
        test_y = torch.split(test_y, batch_size)
        test_X = torch.split(test_X, batch_size)
        

        pred=[]
        label= []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_X):
                predictions = model(batch.float())
                label+=test_y[i].tolist()
                indecies = torch.argmax(predictions, dim =1) 
                pred+=indecies.tolist()
    
        labels = [i for l in label for i in l]
        
        a = accuracy_score(labels,pred)
        p = precision_score(labels,pred, average='weighted')
        f1= f1_score(labels,pred, average='weighted')
        r = recall_score(labels,pred, average='weighted') 
        
        print({"acc":a,"prec":p,"f1":f1,"r":r})
        
        
        pass
