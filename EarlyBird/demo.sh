conda activate kornia 
python -m pdb main.py fit -c configs/t_messytable_fit.yml     -c configs/d_messytable.yml 
python main.py test -c lightning_logs/messytable_center/config.yaml --ckpt lightning_logs/messytable_center/checkpoints/model-epoch=196-val_loss=-18.44.ckpt
