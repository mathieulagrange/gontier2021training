import os
import argparse
import torch
import torch.nn as nn

from model import *
from util import *
import sys
from tqdm import tqdm

def main(config):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    # Load datasets
    dataSpec = np.load(os.path.join(config.data_path, config.dataset+'_spectralData.npy'), mmap_mode='r')
    dataMel = np.load(os.path.join(config.data_path, config.dataset+'_melSpectrograms.npy'), mmap_mode='r')
    dataPres = np.load(os.path.join(config.data_path, config.dataset+'_presence.npy'), mmap_mode='r')
    dataTimePres = np.load(os.path.join(config.data_path, config.dataset+'_time_of_presence.npy'), mmap_mode='r')
    
    if config.exp == 'yamnet':
        sys.path.append('yamnet')
        import yamnet_eval
        presencePath = os.path.join(config.output_path, config.dataset+'_yamnet_presence.npy')
        if not os.path.exists(presencePath) or config.force_recompute:
            score, _ = yamnet_eval.run(dataMel)
            presence = yamnet_eval.score2presence(score)
            np.save(presencePath, presence)
        else:
            presence = np.load(presencePath)
    else:
        settings = load_settings(Path('./exp_settings/', config.exp+'.yaml'))
        modelName = get_model_name(settings)
        print('Model: ', modelName)
        
        presencePath = os.path.join(config.output_path, config.dataset+'_'+modelName+'_presence.npy')
        if not os.path.exists(presencePath) or config.force_recompute:
            useCuda = torch.cuda.is_available() and not settings['training']['force_cpu']
            if useCuda:
                print('Using CUDA.')
                dtype = torch.cuda.FloatTensor
                ltype = torch.cuda.LongTensor
            else:
                print('No CUDA available.')
                dtype = torch.FloatTensor
                ltype = torch.LongTensor
            
            # Model init.
            enc = VectorLatentEncoder(settings)
            dec = PresPredRNN(settings, dtype=dtype)
            if useCuda:
                enc = nn.DataParallel(enc).cuda()
                dec = nn.DataParallel(dec).cuda()
            
            # Pretrained state dict. loading
            enc.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_enc', useCuda=useCuda))
            dec.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_dec', useCuda=useCuda))
            
            print('Encoder: ', enc)
            print('Decoder: ', dec)
            print('Encoder parameter count: ', enc.module.parameter_count() if useCuda else enc.parameter_count())
            print('Decoder parameter count: ', dec.module.parameter_count() if useCuda else dec.parameter_count())
            print('Total parameter count: ', enc.module.parameter_count()+dec.module.parameter_count() if useCuda else enc.parameter_count()+dec.parameter_count())
            
            enc.eval()
            dec.eval()
            presence = np.zeros(dataPres.shape)
            for k in tqdm(range(dataSpec.shape[0])):
                x = torch.Tensor(dataSpec[k,:,:]).type(dtype)
                x = F.pad(x.unsqueeze(0).unsqueeze(0)+settings['data']['level_offset_db'], (0, 3))
                if useCuda:
                    x = x.cuda()
                
                encData = torch.zeros((x.size(0), x.size(2)-7, 128)).type(dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)-7):
                    encData[:, iSeq, :] = enc(x[:, :, iSeq:iSeq+8, :].squeeze(1))
                score = torch.sigmoid(dec(encData))
                presence[k, :, :] = score.squeeze().round().cpu().data
                
            np.save(presencePath, presence)
        else:
            presence = np.load(presencePath)
    
    if config.exp == 'yamnet':
        metricsPath = os.path.join(config.output_path, 'yamnet')
    else:
        metricsPath = os.path.join(config.output_path, modelName)
    # Metrics
    if presence.shape[1] < dataPres.shape[1]: # YAMNet
        reference = dataPres[:, :presence.shape[1]]
    else:
        reference = dataPres
    np.save(metricsPath+'_tppSe.npy', np.mean((np.mean(presence, axis=1)-dataTimePres).flatten()**2))
    
    np.save(metricsPath+'_accuracy.npy', (presence==reference).flatten())
    
    np.save(metricsPath+'_truePositive.npy', np.sum((presence==1) & (reference==1))/np.sum(reference==1))
    np.save(metricsPath+'_trueNegative.npy', np.sum((presence==0) & (reference==0))/np.sum(reference==0))
    np.save(metricsPath+'_falsePositive.npy', np.sum((presence==1) & (reference==0))/np.sum(reference==0))
    np.save(metricsPath+'_falseNegative.npy', np.sum((presence==0) & (reference==1))/np.sum(reference==1))
    print('Overall metrics')
    print(' - Accuracy:             {:.4f}'.format(np.mean((presence==reference).flatten())))
    print(' - True positive rate:   {:.4f}'.format(np.sum((presence==1) & (reference==1))/np.sum(reference==1)))
    print(' - True negative rate:   {:.4f}'.format(np.sum((presence==0) & (reference==0))/np.sum(reference==0)))
    print(' - False positive rate:  {:.4f}'.format(np.sum((presence==1) & (reference==0))/np.sum(reference==0)))
    print(' - False negative rate:  {:.4f}'.format(np.sum((presence==0) & (reference==1))/np.sum(reference==1)))
    print(' - Time of presence MSE: {:.4f}'.format(np.mean((np.mean(presence, axis=1)-dataTimePres).flatten()**2)))
    sources = ['traffic', 'voice', 'bird']
    
    tp_s = np.sum(np.sum((presence==1) & (reference==1), axis=0), axis=0)/np.sum(np.sum(reference==1, axis=0), axis=0)
    tn_s = np.sum(np.sum((presence==0) & (reference==0), axis=0), axis=0)/np.sum(np.sum(reference==0, axis=0), axis=0)
    fp_s = np.sum(np.sum((presence==1) & (reference==0), axis=0), axis=0)/np.sum(np.sum(reference==0, axis=0), axis=0)
    fn_s = np.sum(np.sum((presence==0) & (reference==1), axis=0), axis=0)/np.sum(np.sum(reference==1, axis=0), axis=0)
    act_s = np.mean(np.mean(reference==1, axis=0), axis=0)
    
    for si, s in enumerate(sources):
        print('Metrics for source {}'.format(s))
        np.save(metricsPath+'_'+s+'Accuracy.npy', (presence[:, :, si]==reference[:, :, si]).flatten()) # Normal accuracy
        np.save(metricsPath+'_'+s+'TppSe.npy', (np.mean(presence[:, :, si], axis=1)-dataTimePres[:, si])**2)
        # Source specific accuracies, no conf. weighting implemented
        np.save(metricsPath+'_'+s+'TruePositive.npy',tp_s[si])
        np.save(metricsPath+'_'+s+'TrueNegative.npy',tn_s[si])
        np.save(metricsPath+'_'+s+'FalsePositive.npy',fp_s[si])
        np.save(metricsPath+'_'+s+'FalseNegative.npy',fn_s[si])
        print(' - Accuracy:             {:.4f}'.format(np.mean((presence[:, :, si]==reference[:, :, si]).flatten())))
        print(' - True positive rate:   {:.4f}'.format(tp_s[si]))
        print(' - True negative rate:   {:.4f}'.format(tn_s[si]))
        print(' - False positive rate:  {:.4f}'.format(fp_s[si]))
        print(' - False negative rate:  {:.4f}'.format(fn_s[si]))
        print(' - Time of presence MSE: {:.4f}'.format(np.mean((np.mean(presence[:, :, si], axis=1)-dataTimePres[:, si])**2)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, help='Experience settings YAML, or oracle, chance, null, yamnet')
    parser.add_argument('--dataset', type=str, default='Lorient-1k', help='Evaluation dataset')
    parser.add_argument('--data_path', type=str, default='data', help='Evaluation data path')
    parser.add_argument('--output_path', type=str, default='eval_outputs', help='Evaluation output path')
    parser.add_argument('-force_recompute', action='store_true')
    config = parser.parse_args()
    
    main(config)
