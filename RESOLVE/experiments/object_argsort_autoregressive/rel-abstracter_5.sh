python3 evaluate_argsort_model_learning_curves_1.py --model  rel-abstracter --pretraining_mode "none" --eval_task_data_path "object_sorting_datasets/5/product_structure_reshuffled_object_sort_dataset.npy" --n_epochs 100 --early_stopping True --min_train_size 110 --max_train_size 500 --train_size_step 50 --num_trials 3 --start_trial 0 --pretraining_train_size -1 --wandb_project_name "Sorting-5" 

