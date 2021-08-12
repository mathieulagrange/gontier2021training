import argparse
import os

def main(config):

  inference = [
  'wget -O data/Lorient-1k_spectralData.npy https://zenodo.org/record/4687057/files/Lorient-1k_spectralData.npy?download=1',
  'wget -O data/Lorient-1k_melSpectrograms.npy https://zenodo.org/record/5153616/files/Lorient-1k_melSpectrograms.npy?download=1',
  'wget -O data/Lorient-1k_presence.npy https://zenodo.org/record/4687057/files/Lorient-1k_presence.py?download=1',
  'wget -O data/Lorient-1k_time_of_presence.npy https://zenodo.org/record/4687057/files/Lorient-1k_time_of_presence.py?download=1'
  ]

  training = [
  'wget -O data/CENSE-2k_train_annotations.npy https://zenodo.org/record/4694522/files/CENSE-2k_train_annotations.npy?download=1',
  'wget -O data/CENSE-2k_train_spectralData.npy https://zenodo.org/record/4694522/files/CENSE-2k_train_spectralData.npy?download=1',
  'wget -O data/CENSE-2k_validation_annotations.npy https://zenodo.org/record/4694522/files/CENSE-2k_validation_annotations.npy?download=1', 'wget -O data/CENSE-2k_validation_spectralData.npy https://zenodo.org/record/4694522/files/CENSE-2k_validation_spectralData.npy?download=1',
  'wget -O data/simFSD-18k_train_presence.npy https://zenodo.org/record/4733698/files/simFSD-18k_training_presence.npy?download=1',
  'wget -O data/simFSD-18k_train_spectralData.npy https://zenodo.org/record/4733698/files/simFSD-18k_training_spectralData.npy?download=1', 'wget -O data/simFSD-18k_validation_presence.npy https://zenodo.org/record/4733698/files/simFSD-18k_validation_presence.npy?download=1', 'wget -O data/simFSD-18k_validation_spectralData.npy https://zenodo.org/record/4733698/files/simFSD-18k_validation_spectralData.npy?download=1',
  'wget -O data/FSD-2k_train_presence.npy https://zenodo.org/record/4730390/files/FSD-2k_train_presence.npy?download=1',
  'wget -O data/FSD-2k_train_spectralData.npy https://zenodo.org/record/4730390/files/FSD-2k_train_spectralData.npy?download=1',
  'wget -O data/FSD-2k_validation_presence.npy https://zenodo.org/record/4730390/files/FSD-2k_validation_presence.npy?download=1',
  'wget -O data/FSD-2k_validation_spectralData.npy https://zenodo.org/record/4730390/files/FSD-2k_validation_spectralData.npy?download=1',
  'wget -O data/augCENSE-18k_train_presence.npy https://zenodo.org/record/4733681/files/augCENSE-18k_train_presence.npy?download=1',
  'wget -O data/augCENSE-18k_train_spectralData.npy https://zenodo.org/record/4733681/files/augCENSE-18k_train_spectralData.npy?download=1',
  'wget -O data/augCENSE-18k_validation_presence.npy https://zenodo.org/record/4733681/files/augCENSE-18k_validation_presence.npy?download=1',
  'wget -O data/augCENSE-18k_validation_spectralData.npy https://zenodo.org/record/4733681/files/augCENSE-18k_validation_spectralData.npy?download=1',
  'wget -O data/simCENSE-18k_train_presence.npy https://zenodo.org/record/4694524/files/simCENSE-18k_train_annotations.npy?download=1',  'wget -O data/simCENSE-18k_train_spectralData.npy https://zenodo.org/record/4694524/files/simCENSE-18k_train_spectralData.npy?download=1', 'wget -O data/simCENSE-18k_validation_presence.npy https://zenodo.org/record/4694524/files/simCENSE-18k_validation_annotations.npy?download=1',
  'wget -O data/simCENSE-18k_validation_spectralData.npy https://zenodo.org/record/4694524/files/simCENSE-18k_validation_spectralData.npy?download=1'
  ]

  pretext = [
  'wget -O data/CENSEgram-5M.h5 https://zenodo.org/record/4687030/files/CENSEgram-5M.h5?download=1'
  ]

  if config.task == 'inference':
    datasets = inference
  elif config.task == 'training':
    datasets = training
  elif config.task == 'pretext':
    datasets = pretext

  for dataset in datasets:
    os.system(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='inference', help='Download the datasets for the task: evaluation, training, pretext')
    config = parser.parse_args()

    main(config)
