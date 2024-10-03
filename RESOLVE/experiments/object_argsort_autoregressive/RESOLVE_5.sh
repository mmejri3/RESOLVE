python3 evaluate_argsort_model_learning_curves_1.py --model LARS-VSA --pretraining_mode "none" --eval_task_data_path "object_sorting_datasets/5/product_structure_reshuffled_object_sort_dataset.npy" --n_epochs 300 --early_stopping True --min_train_size 10 --max_train_size 300 --train_size_step 30 --num_trials 3 --start_trial 0 --pretraining_train_size -1 --wandb_project_name "Sorting-5" 

