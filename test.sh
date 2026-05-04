export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

#python train_sb3.py ocr=slotcontrast pooling=transformer sb3=ppo sb3_acnet=mlp env=vizdoom_sc env.checkpoint_path=/pretrained_encoders/slotcontrast_vizdoom.ckpt wandb.offline=True
#python train_sb3.py ocr=slotcontrast pooling=transformer sb3=ppo sb3_acnet=mlp env=maniskill_sc env.checkpoint_path=slotcontrast_maniskill.ckpt wandb.offline=True
python train_sb3.py ocr=dinosaur pooling=transformer sb3=ppo sb3_acnet=mlp env=robosuite_dino wandb.offline=True