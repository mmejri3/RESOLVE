names=("comparison__kth_biggest")
for name in "${names[@]}"
do
    for train_size in 100 1000 10000
    do
	CUDA_VISIBLE_DEVICES=1 python train_model.py --model attentional_lstm --task "$name" --model_size small --run_name "Attentional LSTM $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES"

	    CUDA_VISIBLE_DEVICES=0 python train_model.py --model transformer --task "$name" --model_size small --run_name "Transformer small $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES" 
    done
done

