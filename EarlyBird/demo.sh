conda activate kornia 

#Messytable
#center
python -m pdb main.py fit -c configs/t_messytable_fit.yml     -c configs/d_messytable.yml 
python main.py test -c lightning_logs/messytable_center/config.yaml --ckpt lightning_logs/messytable_center/checkpoints/model-epoch=196-val_loss=-18.44.ckpt

#foot
python main.py fit -c configs/t_messytable_fit.yml -c configs/d_messytable_foot.yml
python main.py test -c lightning_logs/messytable_foot/config.yaml --ckpt lightning_logs/messytable_foot/checkpoints/model-epoch=198-val_loss=-18.34.ckpt

#Synthretail
#center
python -m pdb main.py fit -c configs/t_synthretail_fit.yml     -c configs/d_synthretail.yml 
python main.py test -c lightning_logs/synthretail_center/config.yaml --ckpt lightning_logs/synthretail_center/checkpoints/model-epoch=

