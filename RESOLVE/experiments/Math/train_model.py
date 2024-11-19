'''import warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress TensorFlow warnings
# Suppress Python warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm, trange
import argparse
import os
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True
# load .env env variables (specified TFDS_DATA_DIR)
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import models
#import keras_nlp
import sys; sys.path.append('../'); sys.path.append('../..')
from transformer_modules import TeacherForcingAccuracy
from eval_utils import evaluate_seq2seq_model, log_to_wandb
import utils
os.environ['TFDS_DATA_DIR'] = './'
# region SETUP

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=tuple(models.model_creator_dict.keys()))
parser.add_argument('--model_size', type=str, default='medium')
parser.add_argument('--task', type=str)
parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs to train each model for')
parser.add_argument('--train_size', default=-1, type=int, help='size of training set to take')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
parser.add_argument('--wandb_project_name', default=None, type=str, help='W&B project name')
parser.add_argument('--run_name', default=None, type=str, help='run name')
parser.add_argument('--ignore_gpu_assert', action="store_true", help='whether to confirm that there is a recognized gpu')
parser.add_argument('--seed', default=314159, help='random seed')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())

# set tensorflow random seed
tf.random.set_seed(args.seed)


# set up W&B logging
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb0")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name
if wandb_project_name is None:
    wandb_project_name = f'math-{args.task}'

# timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# model_checkpoints_dir = f'model_checkpoints/{args.task}_{args.model}_{timestamp}'
# os.mkdir(model_checkpoints_dir)

class MaxMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_name='val_teacher_forcing_accuracy'):
        super(MaxMetricsCallback, self).__init__()
        self.metric_name = metric_name
        self.max_metric = -float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # Update the maximum metric value if the current is better
        current_metric = logs.get(self.metric_name, -float('inf'))
        if current_metric > self.max_metric:
            self.max_metric = current_metric
        
    def on_train_end(self, logs=None):
        # Log the maximum metric value to wandb at the end of training
        wandb.log({f'max_{self.metric_name}': self.max_metric})

def create_callbacks(include_max_metric=True):
    callbacks = [
        wandb.keras.WandbCallback(
            monitor="val_loss", verbose=0, mode="auto", save_weights_only=False,
            log_weights=True, log_gradients=False, save_model=False,
            log_batch_frequency=1, log_best_prefix="best_", save_graph=False,
            compute_flops=True)
    ]
    if include_max_metric:
        callbacks.append(MaxMetricsCallback(metric_name='val_teacher_forcing_accuracy'))
    
    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='auto', restore_best_weights=True))
    
    return callbacks


fit_kwargs = {'epochs': args.n_epochs}


#region Dataset
train_examples, val_examples = tfds.load(
    f'math_dataset/{args.task}',
    split=['train', 'test'],
    as_supervised=True)

# global max lengths of questions and answers
max_q_length, max_a_length = 160, 10

start_char = '@'
eos_char = ';'


q_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_q_length,
)

a_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_a_length,
)


def tokens_to_text(id_to_word, tokens):
  words = id_to_word(tokens)
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  result = tf.strings.regex_replace(result, '^ *@ *', '')
  result = tf.strings.regex_replace(result, ' *; *$', '')
  return result
         
    
def prepend_start_token(q,a):
    source = q
    return q, a

source_len = max_q_length
target_len = max_a_length  # max length + 2 (for start token and end token) - 1 ([:-1])
label_len = max_a_length  # max length + 2 (for start token and end token) - 1 ([1:])

train_examples = train_examples.map(prepend_start_token)
val_examples = val_examples.map(prepend_start_token)


q_text_vectorizer.adapt(train_examples.take(500).map(lambda q,a: q))
input_vocab_size = q_text_vectorizer.vocabulary_size()

a_text_vectorizer.adapt(train_examples.take(500).map(lambda q,a:a))
target_vocab_size = a_text_vectorizer.vocabulary_size()

print(a_text_vectorizer.get_vocabulary())
print(target_vocab_size,input_vocab_size)

def vectorize_qa(q,a):
    return q_text_vectorizer(q), a_text_vectorizer(a)

def get_source_target_label(q,a):
    source = q
    target = a
    label = a
    source = tf.ensure_shape(source, (source_len,))
    target = tf.ensure_shape(target, (target_len,))
    label = tf.ensure_shape(label, (label_len,))

    return (source, target), label
def get_max_tokens(train_ds):
    list_max = []
    for (context,target), label in train_ds:
        mask = tf.cast(target!=0, dtype=target.dtype)
        mask_sum = tf.reduce_sum(mask,axis=-1)
        max_num = tf.reduce_max(mask_sum).numpy()
        list_max.append(max_num)
    list_max = np.array(list_max)
    return list_max.max()


def rouge_score_e(dataset, model):
    rouge_l_metric = keras_nlp.metrics.RougeL()
    rouge_1_metric = keras_nlp.metrics.RougeN(order=1)
    rouge_2_metric = keras_nlp.metrics.RougeN(order=2)
    id_to_word = tf.keras.layers.StringLookup(vocabulary = a_text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True) 
    for elements in dataset:
        (context, target), label  = elements
        label = tf.where(label == word_to_id(eos_char).numpy(), 0, label)
        pred_tokens = model.translate(context,max_length=max_a_length+1)  
        pred_sentence = tokens_to_text(id_to_word, pred_tokens)
        true_sentence = tokens_to_text(id_to_word, label) 
        rouge_l_metric.update_state(true_sentence, pred_sentence)
        rouge_1_metric.update_state(true_sentence, pred_sentence)
        rouge_2_metric.update_state(true_sentence, pred_sentence)
    rouge_l_score = rouge_l_metric.result()
    rouge_1_score = rouge_1_metric.result()
    rouge_2_score = rouge_2_metric.result()
    return rouge_l_score, rouge_1_score, rouge_2_score
    
train_examples = train_examples.map(vectorize_qa).map(get_source_target_label)
val_examples = val_examples.map(vectorize_qa).map(get_source_target_label)     

batch_size = args.batch_size
buffer_size = 16_000
val_size = 1000

shuffled_dataset = train_examples.shuffle(buffer_size)
train_ds = shuffled_dataset.take(args.train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_ds = train_examples.shuffle(buffer_size).take(args.train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = shuffled_dataset.skip(args.train_size).take(val_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_ds = val_examples.cache().batch(2*batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = val_examples.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#endregion

def masked_loss_rnn(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(y_true != 0, loss.dtype)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    
    
def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    mask = tf.cast(y_true != 0, y_true.dtype)
    y_pred = y_pred * mask
    sequence_match = tf.reduce_all(tf.equal(y_true, y_pred), axis=-1)
    sequence_match = tf.cast(sequence_match, tf.float32)
    return tf.reduce_mean(sequence_match)

    
#region build model
ignore_class = a_text_vectorizer.get_vocabulary().index('')
assert (ignore_class == q_text_vectorizer.get_vocabulary().index(''))
# loss does not ignore null string
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None)
#create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=4e-3, beta_1=0.9, beta_2=0.995, epsilon=1e-9)
create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9)
#create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=1e-3, beta_1=0.92, beta_2=0.995, epsilon=1e-9)
# teacher-forcing acc ignores null string
teacher_forcing_accuracy = TeacherForcingAccuracy(ignore_class=ignore_class)

word_to_id = tf.keras.layers.StringLookup(vocabulary=a_text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]', name='word_to_id')



model = models.model_creator_dict[args.model](input_vocab_size, target_vocab_size, size=args.model_size, word_to_id=word_to_id)

model.compile(loss=loss, optimizer=create_opt(), metrics=masked_acc)
model(next(iter(train_ds.take(1)))[0]); # build model
print(model.summary())


    
            
model.fit(train_ds, validation_data=val_ds, epochs=args.n_epochs, validation_freq=1, verbose=1)'''





import warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress TensorFlow warnings
# Suppress Python warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm, trange
import argparse
import os
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True
# load .env env variables (specified TFDS_DATA_DIR)
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import models
#import keras_nlp
import sys; sys.path.append('../'); sys.path.append('../..')
from transformer_modules import TeacherForcingAccuracy
from eval_utils import evaluate_seq2seq_model, log_to_wandb
import utils
os.environ['TFDS_DATA_DIR'] = './'
# region SETUP

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=tuple(models.model_creator_dict.keys()))
parser.add_argument('--model_size', type=str, default='medium')
parser.add_argument('--task', type=str)
parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs to train each model for')
parser.add_argument('--train_size', default=-1, type=int, help='size of training set to take')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
parser.add_argument('--wandb_project_name', default=None, type=str, help='W&B project name')
parser.add_argument('--run_name', default=None, type=str, help='run name')
parser.add_argument('--ignore_gpu_assert', action="store_true", help='whether to confirm that there is a recognized gpu')
parser.add_argument('--seed', type=int, default=314159, help='random seed')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())

# set tensorflow random seed
tf.random.set_seed(args.seed)


# set up W&B logging
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb0")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name
if wandb_project_name is None:
    wandb_project_name = f'math-{args.task}'

# timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# model_checkpoints_dir = f'model_checkpoints/{args.task}_{args.model}_{timestamp}'
# os.mkdir(model_checkpoints_dir)

class MaxMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_name='val_teacher_forcing_accuracy'):
        super(MaxMetricsCallback, self).__init__()
        self.metric_name = metric_name
        self.max_metric = -float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # Update the maximum metric value if the current is better
        current_metric = logs.get(self.metric_name, -float('inf'))
        if current_metric > self.max_metric:
            self.max_metric = current_metric
        
    def on_train_end(self, logs=None):
        # Log the maximum metric value to wandb at the end of training
        wandb.log({f'max_{self.metric_name}': self.max_metric})

def create_callbacks(include_max_metric=True):
    callbacks = [
        wandb.keras.WandbCallback(
            monitor="val_loss", verbose=0, mode="auto", save_weights_only=False,
            log_weights=True, log_gradients=False, save_model=False,
            log_batch_frequency=1, log_best_prefix="best_", save_graph=False,
            compute_flops=True)
    ]
    if include_max_metric:
        callbacks.append(MaxMetricsCallback(metric_name='val_teacher_forcing_accuracy'))
    
    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='auto', restore_best_weights=True))
    
    return callbacks


fit_kwargs = {'epochs': args.n_epochs}


#region Dataset
train_examples, val_examples = tfds.load(
    f'math_dataset/{args.task}',
    split=['train', 'test'],
    as_supervised=True)

# global max lengths of questions and answers
max_q_length, max_a_length = 160, 30

start_char = '@'
eos_char = ';'


q_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_q_length,
)

a_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_a_length+2,
)


def tokens_to_text(id_to_word, tokens):
  words = id_to_word(tokens)
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  result = tf.strings.regex_replace(result, '^ *@ *', '')
  result = tf.strings.regex_replace(result, ' *; *$', '')
  return result
         
    
def prepend_start_token(q,a):
    source = q
    a = start_char + a + eos_char
    return q, a

source_len = max_q_length
target_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([:-1])
label_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([1:])

train_examples = train_examples.map(prepend_start_token)
val_examples = val_examples.map(prepend_start_token)


q_text_vectorizer.adapt(train_examples.take(1000).map(lambda q,a: q))
input_vocab_size = q_text_vectorizer.vocabulary_size()

a_text_vectorizer.adapt(train_examples.take(1000).map(lambda q,a:a))
target_vocab_size = a_text_vectorizer.vocabulary_size()

print(a_text_vectorizer.get_vocabulary())
print(target_vocab_size,input_vocab_size)

def vectorize_qa(q,a):
    return q_text_vectorizer(q), a_text_vectorizer(a)

def get_source_target_label(q,a):
    source = q
    target = a[:-1]
    label = a[1:]
    source = tf.ensure_shape(source, (source_len,))
    target = tf.ensure_shape(target, (target_len,))
    label = tf.ensure_shape(label, (label_len,))

    return (source, target), label
def get_max_tokens(train_ds):
    list_max = []
    for (context,target), label in train_ds:
        mask = tf.cast(target!=0, dtype=target.dtype)
        mask_sum = tf.reduce_sum(mask,axis=-1)
        max_num = tf.reduce_max(mask_sum).numpy()
        list_max.append(max_num)
    list_max = np.array(list_max)
    return list_max.max()


def rouge_score_e(dataset, model):
    rouge_l_metric = keras_nlp.metrics.RougeL()
    rouge_1_metric = keras_nlp.metrics.RougeN(order=1)
    rouge_2_metric = keras_nlp.metrics.RougeN(order=2)
    id_to_word = tf.keras.layers.StringLookup(vocabulary = a_text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True) 
    for elements in dataset:
        (context, target), label  = elements
        label = tf.where(label == word_to_id(eos_char).numpy(), 0, label)
        pred_tokens = model.translate(context,max_length=max_a_length+1)  
        pred_sentence = tokens_to_text(id_to_word, pred_tokens)
        true_sentence = tokens_to_text(id_to_word, label) 
        rouge_l_metric.update_state(true_sentence, pred_sentence)
        rouge_1_metric.update_state(true_sentence, pred_sentence)
        rouge_2_metric.update_state(true_sentence, pred_sentence)
    rouge_l_score = rouge_l_metric.result()
    rouge_1_score = rouge_1_metric.result()
    rouge_2_score = rouge_2_metric.result()
    return rouge_l_score, rouge_1_score, rouge_2_score
    
train_examples = train_examples.map(vectorize_qa).map(get_source_target_label)
val_examples = val_examples.map(vectorize_qa).map(get_source_target_label)     

batch_size = args.batch_size
buffer_size = 16_000
val_size = 1000

shuffled_dataset = train_examples.shuffle(buffer_size)
train_ds = shuffled_dataset.take(args.train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_ds = train_examples.shuffle(buffer_size).take(args.train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = shuffled_dataset.skip(args.train_size).take(val_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_ds = val_examples.cache().batch(2*batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = val_examples.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#endregion
        
        
def masked_loss_rnn(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,  reduction='none')
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(y_true != 0, loss.dtype)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
   
    
    
        
    
def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    mask = tf.cast(y_true != 0, y_true.dtype)
    y_pred = y_pred * mask
    sequence_match = tf.reduce_all(tf.equal(y_true, y_pred), axis=-1)
    sequence_match = tf.cast(sequence_match, tf.float32)
    return tf.reduce_mean(sequence_match)
    
#region build model
ignore_class = a_text_vectorizer.get_vocabulary().index('')
assert (ignore_class == q_text_vectorizer.get_vocabulary().index(''))
# loss does not ignore null string
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
#create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=4e-3, beta_1=0.9, beta_2=0.995, epsilon=1e-9)
create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9)
#create_opt = lambda : tf.keras.optimizers.AdamW(learning_rate=1e-3, beta_1=0.92, beta_2=0.995, epsilon=1e-9)
# teacher-forcing acc ignores null string
teacher_forcing_accuracy = TeacherForcingAccuracy(ignore_class=ignore_class)

word_to_id = tf.keras.layers.StringLookup(vocabulary=a_text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]', name='word_to_id')

def full_seq(dataset, model):
    total_matches = 0
    total_elements = 0
    acc=0.0
    c=0.0
    for elements in dataset:
        (context, target), label = elements
        # Generate predictions from the model
        pred_tokens = model.translate(context, max_length=max_a_length+1)
        label = tf.where(label == word_to_id(eos_char).numpy(), 0, label)
        eos_mask = tf.equal(pred_tokens, word_to_id(eos_char))
        # Step 2: Compute the cumulative sum along axis 1 to find positions after 'eos'
        cumsum_mask = tf.cumsum(tf.cast(eos_mask, tf.int32), axis=1)

        # Step 3: Create the final mask: 1 where cumsum_mask is 0, else 0
        final_mask = tf.cast(cumsum_mask == 0, tf.int32)

        # If you want the mask to be of dtype float, you can cast it
        final_mask = tf.cast(final_mask, tf.float32)

        # Compare predicted tokens with the target tokens
        matches = tf.reduce_all(tf.equal(pred_tokens, label), axis=1)

        # Count the number of matches
        count_matches = tf.reduce_sum(tf.cast(matches, tf.int32))

        acc+=count_matches.numpy()
        c+=len(label)
    
    # Calculate the element-wise accuracy
    accuracy = acc / c
    return accuracy




def element_matching(dataset, model):
    total_matches = 0
    total_elements = 0

    for elements in dataset:
        (context, target), label = elements

        # Generate predictions from the model
        pred_tokens = model.translate(context, max_length=max_a_length + 1)
        label = tf.where(label == word_to_id(eos_char).numpy(), 0, label)

        # Create a mask for tokens before 'eos' and exclude zero tokens
        eos_mask = tf.equal(pred_tokens, word_to_id(eos_char))
        cumsum_mask = tf.cumsum(tf.cast(eos_mask, tf.int32), axis=1)
        eos_final_mask = tf.cast(cumsum_mask == 0, tf.float32)  # Exclude tokens after 'eos'

        # Exclude zero tokens
        non_zero_mask = tf.cast(pred_tokens != 0, tf.float32)

        # Combine the masks
        final_mask = eos_final_mask * non_zero_mask

        # Compare predicted tokens with the target tokens
        token_matches = tf.equal(pred_tokens, label)
        token_matches = tf.cast(token_matches, tf.float32) * final_mask  # Apply mask

        # Count the number of matching tokens and total tokens per sequence
        num_matches = tf.reduce_sum(token_matches, axis=1)
        total_tokens = tf.reduce_sum(final_mask, axis=1)

        # Sum across all sequences
        total_matches += tf.reduce_sum(num_matches).numpy()
        total_elements += tf.reduce_sum(total_tokens).numpy()

    # Calculate the element-wise accuracy
    accuracy = total_matches / total_elements if total_elements > 0 else 0.0
    return accuracy


model = models.model_creator_dict[args.model](input_vocab_size, target_vocab_size, size=args.model_size, word_to_id=word_to_id)

model.compile(loss=masked_loss_rnn, optimizer=create_opt(), metrics=masked_acc)
model(next(iter(train_ds.take(1)))[0]); # build model
print(model.summary())
#endregion

#region train model
run = wandb.init(project=wandb_project_name, group=f'{args.model}-{args.model_size}', name=args.run_name,
                 config=vars(args), dir='/storage/home/hcoda1/4/mmejri3/scratch/wandb')



import wandb
import tensorflow as tf

class RougeScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, model, log_interval=1):
        super(RougeScoreCallback, self).__init__()
        self.val_ds = val_ds
        self.model = model
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == self.log_interval - 1:
            r_l, r_1, r_2 = self.rouge_score()
            avg_r_l = self.average_rouge_scores(r_l)
            avg_r_1 = self.average_rouge_scores(r_1)
            avg_r_2 = self.average_rouge_scores(r_2)

            # Log the scores using wandb
            wandb.log({
                'epoch': epoch,
                'rouge_l_precision': avg_r_l['precision'],
                'rouge_l_recall': avg_r_l['recall'],
                'rouge_l_f1_score': avg_r_l['f1_score'],
                'rouge_1_precision': avg_r_1['precision'],
                'rouge_1_recall': avg_r_1['recall'],
                'rouge_1_f1_score': avg_r_1['f1_score'],
                'rouge_2_precision': avg_r_2['precision'],
                'rouge_2_recall': avg_r_2['recall'],
                'rouge_2_f1_score': avg_r_2['f1_score'],
            })

            print(f"    The ROUGE Scores are r_l: {avg_r_l['f1_score']}, r_1: {avg_r_1['f1_score']}, r_2: {avg_r_2['f1_score']}")

    def rouge_score(self):        
        r_l, r_1, r_2 = rouge_score_e(self.val_ds, self.model)
        return r_l, r_1, r_2

    def val_acc_c(self):
        val_acc = loss_val(self.val_ds, self.model)
        return val_acc 

    def average_rouge_scores(self, scores):
        avg_scores = {
            'precision': scores['precision'].numpy(),  # Convert scalar tensor to numpy
            'recall': scores['recall'].numpy(),
            'f1_score': scores['f1_score'].numpy(),
        }
        return avg_scores
        
        
class ElementMatchingAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, model, log_interval=1):
        super(ElementMatchingAccuracyCallback, self).__init__()
        self.val_ds = val_ds
        self.model = model
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == self.log_interval - 1:
            accuracy = element_matching(self.val_ds,self.model)

            # Log the accuracy using wandb
            wandb.log({
                'epoch': epoch,
                'element_matching_accuracy': accuracy,
            })

            print(f"    Element Matching Accuracy at epoch {epoch}: {accuracy:.4f}")


                


def create_callbacks(include_max_metric=True):
    callbacks = [
        wandb.keras.WandbCallback(
            monitor="val_loss", verbose=0, mode="auto", save_weights_only=False,
            log_weights=False, log_gradients=False, save_model=False,
            log_batch_frequency=1, log_best_prefix="best_", save_graph=False,
            compute_flops=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_masked_acc", patience=150, mode='max', restore_best_weights=True)]
        #ElementMatchingAccuracyCallback(val_ds, model, log_interval=args.n_epochs//10)]
    return callbacks
    
            
model.fit(train_ds, validation_data=val_ds, epochs=args.n_epochs, validation_freq=1, verbose=1,  callbacks=create_callbacks())
full_seq_accuracy = full_seq(test_ds,model)
element_wise_accuracy = element_matching(test_ds,model)
evaluation_dict = {'element_wise_acuracy':element_wise_accuracy}
evaluation_dict['full_seq_accuracy'] = full_seq_accuracy
wandb.log(evaluation_dict)

#module_no_signatures_path = f'./models_ckpt/{args.task}/{args.model}/'
#model(next(iter(train_ds.take(1)))[0]); # build model
#print('Saving model...')
#tf.saved_model.save(model, module_no_signatures_path)  
#wandb.finish(quiet=True)
#endregion


