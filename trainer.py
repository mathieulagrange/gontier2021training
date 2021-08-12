import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import math
import csv
import numpy as np
import os

class PresPredTrainer:
    def __init__(self, settings, enc, dec, modelName, devDataset, valDataset=None, dtype=torch.FloatTensor, ltype=torch.LongTensor):
        self.dtype = dtype
        self.ltype = ltype
        
        self.enc = enc
        self.dec = dec
        self.devDataset = devDataset
        self.valDataset = valDataset
        self.tLen = settings['data']['texture_length']
        self.lr = settings['training']['lr']
        self.encOptimizer = optim.Adam(params=self.enc.parameters(), lr=self.lr)
        self.decOptimizer = optim.Adam(params=self.dec.parameters(), lr=self.lr)
        self.modelName = modelName
        self.seqInput = settings['data']['seq_length'] != 1
        
        self.do_validate = settings['workflow']['validate']
        self.optEnc = settings['model']['encoder']['finetune']
        
        self.checkpointDir = settings['model']['checkpoint_dir']
        self.loggingDir = settings['model']['logging_dir']
        self.encEmbeddingSize = settings['model']['encoder']['embedding_size']
        
    def train(self, batchSize=32, epochs=10):
        self.trainDataloader = torch.utils.data.DataLoader(self.devDataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        if self.valDataset is not None:
            self.valDataloader = torch.utils.data.DataLoader(self.valDataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        losses = []
        lossesVal = []
        
        if epochs>0 and self.do_validate:
            lossVal = self.validate(-1)
            lossesVal.append(lossVal.cpu().numpy())
        
        if self.optEnc:
            self.enc.train()
        else:
            self.enc.eval()
        self.dec.train()
        cur_loss = 0
        for currentEpoch in range(epochs):
            # Training
            with tqdm(self.trainDataloader, desc='Epoch {}, loss: {:.4f}'.format(currentEpoch+1, cur_loss)) as t:
                for currentBatch, (x, p) in enumerate(t):
                    x = x.type(self.dtype)
                    p = p.type(self.dtype)
                    
                    if self.seqInput:
                        r = torch.zeros((x.size(0), x.size(2)-(self.tLen-1), self.encEmbeddingSize)).type(self.dtype) # batch x seq_len x embedding_size
                        for iSeq in range(x.size(2)-(self.tLen-1)):
                            r[:, iSeq, :] = self.enc(x[:, :, iSeq:iSeq+self.tLen, :].squeeze(1))
                    else:
                        r = self.enc(x.squeeze(1))
                    o = self.dec(r)
                    
                    loss = F.binary_cross_entropy_with_logits(o, p.squeeze(1))
                    
                    if self.optEnc:
                        self.encOptimizer.zero_grad()
                    self.decOptimizer.zero_grad()
                    loss.backward()
                    
                    #tqdm.write('Loss is {:.4f}'.format(loss.data))
                    cur_loss = loss.data.cpu().numpy()
                    t.set_description('Epoch {}, loss: {:.4f}'.format(currentEpoch+1, cur_loss))
                    losses.append(cur_loss)
                    
                    if self.optEnc:
                        self.encOptimizer.step()
                    self.decOptimizer.step()
            
            # Validation
            if self.do_validate:
                lossVal = self.validate(currentEpoch)
                lossesVal.append(lossVal.cpu().numpy())
            
            # Save model state
            if not os.path.exists(self.checkpointDir):
                os.makedirs(self.checkpointDir)
            torch.save(self.enc.state_dict(), os.path.join(self.checkpointDir, 'model_' + self.modelName + '_enc_Epoch' + str(currentEpoch+1) + '.pt'))
            torch.save(self.dec.state_dict(), os.path.join(self.checkpointDir, 'model_' + self.modelName + '_dec_Epoch' + str(currentEpoch+1) + '.pt'))
            
        if epochs>0:
            if not os.path.exists(self.loggingDir):
                os.makedirs(self.loggingDir)
            with open(os.path.join(self.loggingDir, 'loss_'+self.modelName+'.txt'), 'w') as lf:
                writer = csv.writer(lf)
                for l in losses:
                    writer.writerow([np.round(l*10000)/10000])
            with open(os.path.join(self.loggingDir, 'lossVal_'+self.modelName+'.txt'), 'w') as lf:
                writer = csv.writer(lf)
                for l in lossesVal:
                    writer.writerow([np.round(l*10000)/10000])
    
    def validate(self, currentEpoch):
        self.enc.eval()
        self.dec.eval()
        lossVal = 0
        for currentBatch, (x, p) in enumerate(tqdm(self.valDataloader, desc='Epoch {}'.format(currentEpoch+1))):
            x = x.type(self.dtype)
            p = p.type(self.dtype)
            
            if self.seqInput:
                r = torch.zeros((x.size(0), x.size(2)-(self.tLen-1), self.encEmbeddingSize)).type(self.dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)-(self.tLen-1)):
                    r[:, iSeq, :] = self.enc(x[:, :, iSeq:iSeq+self.tLen, :].squeeze())
            else:
                r = self.enc(x.squeeze())
            o = self.dec(r)
            
            loss = F.binary_cross_entropy_with_logits(o, torch.squeeze(p))
            
            lossVal = lossVal + loss.data
        lossVal = lossVal/len(self.valDataloader)
        print(" => Validation loss at epoch {} is {:.4f}".format(currentEpoch, lossVal))
        
        if self.optEnc:
            self.enc.train()
        self.dec.train()
        
        return lossVal
        
