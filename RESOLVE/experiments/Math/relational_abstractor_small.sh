names=("numbers__base_conversion")
for name in "${names[@]}"
do
    for train_size in  100 1000 10000
    do
	CUDA_VISIBLE_DEVICES=0 python train_model.py --model relational_abstractor --task "$name" --model_size small --run_name "rel-abst-small $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES"

        CUDA_VISIBLE_DEVICES=0 python train_model.py --model symbolic_abstractor --task "$name" --model_size small --run_name "symb-abst-small $train_size" --n_epochs 1000 --batch_size 128 --train_size $train_size --wandb_project_name "${name}-ICLR-LAST-ES"
    done
done

