import os
import argparse
import torch
import torch.nn as nn

from model import *
from data_loader import PresPredDataset
from trainer import PresPredTrainer
from util import *

def main(config):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    settings = load_settings(Path('./exp_settings/', config.exp+'.yaml'))

    useCuda = torch.cuda.is_available() and not settings['training']['force_cpu']
    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor

    modelName = get_model_name(settings)

    # Load datasets
    devDataset = PresPredDataset(settings['data'], subset='train')
    if settings['workflow']['validate']:
        valDataset = PresPredDataset(settings['data'], subset='validation')
    else:
        valDataset = None
    
    # Encoder init.
    enc = VectorLatentEncoder(settings)

    # Decision init.
    if settings['model']['classifier']['type'] == 'rnn':
        dec = PresPredRNN(settings, dtype=dtype)
    else:
        raise NotImplementedError

    if useCuda:
        enc = nn.DataParallel(enc).cuda()
        dec = nn.DataParallel(dec).cuda()

    # Pretrained state dict loading
    if settings['model']['encoder']['pretraining'] is not None:
        loadedEncoder = load_latest_model_from(settings['model']['encoder']['pretrained_dir'], settings['model']['encoder']['pretrained_checkpoint'], useCuda=useCuda)
        enc.load_state_dict(loadedEncoder)
    if settings['model']['load_full_pretrained']:
        enc.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_enc', useCuda=useCuda))
        dec.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_dec', useCuda=useCuda))

    print('Model: ', modelName)
    print('Encoder: ', enc)
    print('Decoder: ', dec)
    print('Encoder parameter count: ', enc.module.parameter_count() if useCuda else enc.parameter_count())
    print('Decoder parameter count: ', dec.module.parameter_count() if useCuda else dec.parameter_count())
    print('Total parameter count: ', enc.module.parameter_count()+dec.module.parameter_count() if useCuda else enc.parameter_count()+dec.parameter_count())

    trainer = PresPredTrainer(settings, enc, dec, modelName, devDataset, valDataset=valDataset, dtype=dtype, ltype=ltype)

    if settings['workflow']['train']:
        trainer.train(batchSize=settings['training']['batch_size'], epochs=settings['training']['nb_epochs'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='exp001', help='Experience settings YAML')
    config = parser.parse_args()

    main(config)
