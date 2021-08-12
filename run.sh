echo 'Setting up environnement'

python3.7 -m venv env/
source env/bin/activate
pip install -r requirements.txt

echo 'Evaluation'

python download.py --task inference
python inference.py --exp yamnet
python inference.py --exp FSD-2k_CENSEgram-5M
python inference.py --exp simFSD-18k_CENSEgram-5M
python inference.py --exp simCENSE-18k_CENSEgram-5M
python inference.py --exp simCENSE-18k_scratch
python inference.py --exp augCENSE-18k_CENSEgram-5M

echo 'Training'

python download.py --task training
python train.py --exp FSD-2k_CENSEgram-5M
python train.py --exp simFSD-18k_CENSEgram-5M
python train.py --exp simCENSE-18k_CENSEgram-5M
python train.py --exp simCENSE-18k_scratch
python train.py --exp augCENSE-18k_CENSEgram-5M

echo 'Prextext training'

python download.py --task pretext
