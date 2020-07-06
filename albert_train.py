# -*- encoding: utf-8 -*-
'''
@File    :   albert_train.py
@Time    :   2020/07/05 21:23:19
@Author  :   Luo Jianhui 
@Version :   1.0
@Contact :   kid1412ljh@outlook.com
'''

# here put the import lib
from albert_reg_model import AlbertForSequenceRegression
from transformers import AdamW, AlbertConfig
from transformers import get_linear_schedule_with_warmup
from data_preprocess import PreProcess
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from early_stopping import EarlyStopping

import time
import datetime
import torch
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr


def dataloader(path):
    
    train_dataset, val_dataset = PreProcess(path).split_data()
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(
            val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


def get_model(path, epochs):
    train_dataloader, _ = dataloader(path)

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Load BertForTokenClassification, the pretrained BERT model with a single, linear classification layer on top.
    model = AlbertForSequenceRegression.from_pretrained(
        'albert-base-v2',  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=30,  # The number of output labels--2 for binary classification. 
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,  # args.learning_rate - default is 5e-5, 
        eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    )
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps)

    return model, optimizer, scheduler


def spearman(preds, targets):
    target_len = 30
    preds = preds.reshape(-1, target_len)
    targets = targets.reshape(-1, target_len)
    score = 0
    for i in range(len(target_len)):
        print(i, spearmanr(targets[:,i], preds[:,i]))
        score += np.nan_to_num(spearmanr(targets[:, i], preds[:, i]).correlation)
    return score/target_len


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit(path, epochs=10):
    """
    Args:
        epochs: Number of training epochs. The BERT authors recommend between 2 and 4.
                We chose to run for 4, but we'll see later that this may be over-fitting the training data.
    """

    train_dataloader, validation_dataloader = dataloader(path)
    model, optimizer, scheduler = get_model(path, epochs)

    early_stopping = EarlyStopping()
    # Tell pytorch to run this model on the GPU.
    # model.cuda()
    model.cpu()
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(tqdm(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: token_type_ids
            #   [2]: attention masks
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_token_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 token_type_ids=b_token_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: token_type_ids
            #   [2]: attention masks
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_token_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=b_token_ids,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the spearman for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += spearman(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Add early stopping
        early_stopping(avg_val_accuracy, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time() - total_t0)))

if __name__ == '__main__':
    path = 'google-quest-challenge/train.csv'
    fit(path, epochs=4)