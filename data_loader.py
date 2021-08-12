import os
import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F

class PresPredDataset(torch.utils.data.Dataset):
    def __init__(self, settings, subset='train'):
        self.datasetName = settings['dataset_name']
        self.datasetPath = os.path.join(settings['root_dir'], self.datasetName)
        self.sr = settings['sr']
        self.tLen = settings['texture_length']
        self.fLen = settings['frame_length']
        self.hLen = settings['hop_length']
        self.lvlOffset = settings['level_offset_db']
        self.seqLen = settings['seq_length']
        self.classes = settings['classes']
        self.nClasses = len(self.classes)

        self.data_tob = np.load(self.datasetPath+'_'+subset+'_spectralData.npy', mmap_mode='r')
        self.data_pres = np.load(self.datasetPath+'_'+subset+'_presence.npy', mmap_mode='r')

        self.nFrames = self.data_pres.shape[1]
        self.allowExampleOverlap = settings['allow_example_overlap']

        print('{} dataset {} split length: {}'.format(self.datasetName, subset, self.__len__()))

    def __getitem__(self, idx):
        iFile = int(np.floor(idx/int(np.floor(self.nFrames/self.seqLen))))
        iEx = np.mod(idx, int(np.floor(self.nFrames/self.seqLen))) # Index of exemple within file
        input_x = torch.unsqueeze(torch.from_numpy(self.data_tob[iFile, iEx*self.seqLen:(iEx+1)*self.seqLen+(self.tLen-1), :]), 0)
        pres = torch.unsqueeze(torch.from_numpy(self.data_pres[iFile, iEx*self.seqLen:(iEx+1)*self.seqLen, :]), 0)
        return F.pad(input_x+self.lvlOffset, (0, 3)), pres # Pad last dimension (freq) from 29 to 32 for the encoder, plus level correction for simulated data

    def __len__(self):
        return self.data_pres.shape[0]*int(np.floor(self.nFrames/self.seqLen))
