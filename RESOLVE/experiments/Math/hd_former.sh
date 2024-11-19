names=("comparison__closest")
for train_size in 100 1000 10000; do
        for name in "${names[@]}"; do
	CUDA_VISIBLE_DEVICES=0 python train_model.py --model hdformer --task "$name" --model_size small --run_name "hd_former-small No Enc Symb False $train_size" --n_epochs 1000 --batch_size 128 --train_size  $train_size  --wandb_project_name "${name}-TMLR-New"
    done
done
