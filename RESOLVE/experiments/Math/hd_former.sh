<<<<<<< HEAD
names=("comparison__closest")
for train_size in 100 1000 10000; do
        for name in "${names[@]}"; do
	CUDA_VISIBLE_DEVICES=0 python train_model.py --model hdformer --task "$name" --model_size small --run_name "hd_former-small No Enc Symb False $train_size" --n_epochs 1000 --batch_size 128 --train_size  $train_size  --wandb_project_name "${name}-TMLR-New"
=======
names=("numbers__place_value")
for name in "${names[@]}"
do
    for train_size in  100
    do


	#CUDA_VISIBLE_DEVICES=0 python train_model.py --model hdformer --task "$name" --model_size small --run_name "HD Former small $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES-988"

	CUDA_VISIBLE_DEVICES=0 python train_model.py --model transformer --task "$name" --model_size small --run_name "transformer-small $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES-988"

	#CUDA_VISIBLE_DEVICES=0 python train_model.py --model attentional_lstm --task "$name" --model_size small --run_name "Attentional LSTM $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES"
>>>>>>> b99bceb5b9fe9b94ff47fe8ffbf37580e586a9e3
    done
done

