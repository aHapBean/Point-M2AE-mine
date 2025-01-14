CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name test

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_pb.yaml --finetune_model --exp_name test --ckpts experiments/pre-training/point-m2ae/test/ckpt-best.pth