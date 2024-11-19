import numpy as np
from tqdm import tqdm
from transformer_modules import TeacherForcingAccuracy
import wandb

teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)

def evaluate_seq2seq_model(model, val_ds, start_token, eos_token, print_=False):
    elementwise_acc, acc_per_position, seq_acc = 0.0, 0.0, 0.0
    
    for step, val in tqdm(enumerate(val_ds)):
        (source_test, target_test), labels_test = val
        n, seqs_length = np.shape(target_test)

        # Initialize the output array with the start token
        output = np.full((n, seqs_length + 1), start_token, dtype=int)

        # Create a mask to check if each sequence has reached the EOS token
        eos_mask = np.zeros(n, dtype=bool)

        for i in range(seqs_length):
            # Make predictions for the entire batch at once
            predictions = model((source_test, output[:, :-1]), training=False)
            predicted_id = np.argmax(predictions[:, i, :], axis=-1)
            output[:, i + 1] = predicted_id

            # Update EOS mask
            eos_mask |= (predicted_id == eos_token)

            # If all sequences in the batch have reached EOS, break early
            if np.all(eos_mask):
                break

        # Calculate accuracies
        elementwise_acc += np.mean(output[:, 1:] == labels_test)
        acc_per_position += np.array([np.mean(output[:, i+1] == labels_test[:, i]) for i in range(seqs_length)])
        seq_acc += np.mean(np.all(output[:, 1:] == labels_test, axis=1))


    #teacher_forcing_acc = teacher_forcing_acc_metric(labels_test, model([source_test, target_test]))
    #teacher_forcing_acc_metric.reset_state()

    print('element-wise accuracy: %.2f%%' % (100*elementwise_acc/step))
    print('full sequence accuracy: %.2f%%' % (100*seq_acc/step))
    #print('teacher-forcing accuracy:  %.2f%%' % (100*teacher_forcing_acc))



    return None

def log_to_wandb(model, evaluation_dict):
    acc_by_position_table = wandb.Table(
        data=[(i, acc) for i, acc in enumerate(evaluation_dict['acc_by_position'])], 
        columns=["position", "element-wise accuracy at position"])

    evaluation_dict['acc_by_position'] = wandb.plot.line(
        acc_by_position_table, "position", "element-wise accuracy at position",
        title="Element-wise Accuracy By Position")

    wandb.log(evaluation_dict)
