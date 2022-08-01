"""
Builds models 
"""
import time
import datetime
import random
import numpy as np
import os
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertForTokenClassification, AdamW, CamembertConfig, CamembertTokenizer
from transformers import BertForSequenceClassification, BertForTokenClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME




class BertClassifier():
    def __init__(self, config=None, bert_model=None):
        """
        Requires either a config, or manually-passed arguments following the data config template
        """
        if config is not None:
            self.data = config.data
            self.modelargs = config.model
            self.train = config.train
        if bert_model is not None:
            self.bert_model = bert_model
        else:
            self.bert_model = "CamemBert"

    def fit_evaluate(self, train, test, **kwargs):
        """
        New option : "only_eval_last" = evaluation only at last epoch, in order to avoid using too much ressources

        """
        if not hasattr(self, 'model') or self.modelargs=={}:
            self.modelargs = kwargs
        
        # Check for params presence & define them
        batch_size = self.modelargs['batch_size']
        sampler = self.modelargs['sampler']
        epochs = self.modelargs['nepochs']
        if 'only_eval_last' in self.modelargs.keys() and self.modelargs['only_eval_last'] == True:
            only_eval_last = True
        else:
            only_eval_last = False

        try:
            seed_val = self.modelargs['random_seed']
        except:
            seed_val = 42
        try:
            lr = self.modelargs['learning_rate']
        except:
            lr = 5e-5
        try:
            eps = self.modelargs['eps']
        except:
            eps = 1e-8


        # Decompose train and test as subvectors
        train_inputs, train_labels, train_masks = train
        validation_inputs, validation_labels, validation_masks = test

        # Put train & test in tensors
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        # Need for nb of labels. numpy unique automatically flattens when needed
        nb_labels = len(np.unique(np.concatenate([train_labels, validation_labels])))

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        if sampler=='sequential':
            train_sampler = SequentialSampler(train_data)
        elif sampler=='random':
            train_sampler = RandomSampler(train_data)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler,
                                    batch_size=batch_size)

        # Create the DataLoader for our validation set.
        # len of val dataset
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        # Model: 
        if self.bert_model == "CamemBert":
            model = CamembertForSequenceClassification.from_pretrained(
                    "camembert-base",
                    num_labels = nb_labels, 
                    output_attentions = False,
                    output_hidden_states = False,
                )
        elif self.bert_model == "Bert":
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased",
                num_labels = nb_labels, 
                output_attentions = False,
                output_hidden_states = False,
                )
        else:
            raise NotImplementedError
        # TODO : pass cuda and device AS A PARAMETER
        device = torch.device("cuda")
        model.cuda()
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                        lr = lr,
                        eps = eps 
                        )

        # Number of training epochs (authors recommend between 2 and 4)
        #epochs = 3

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []
        # For each epoch...
        best_score = 0
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            t0 = time.time()
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            # Reset the total loss for this epoch.
            total_loss = 0

            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()

                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels) 
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            if only_eval_last == False or epoch_i == epochs-1:
                print("")
                print("Running Validation...")
                t0 = time.time()
                model.eval()

                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                true=[]
                pred = []
                # Evaluate data for one epoch
                for batch in validation_dataloader:

                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)

                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch

                    # Telling the model not to compute or store gradients, saving memory and
                    # speeding up validation
                    with torch.no_grad():        
                        outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask)

                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    logits = outputs[0]

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    lab = np.argmax(logits, axis=1).flatten()
                    true_lab = label_ids.flatten()
                    true = np.concatenate([true, true_lab])
                    pred = np.concatenate([pred, lab])
                print(classification_report(true, pred))
                #print(classification_report(true, pred))
                perfs  = precision_recall_fscore_support(true, pred)
                if perfs[2].mean() > best_score:
                    best_perfs = perfs
                    best_epoch = epoch_i
                    best_score = perfs[2].mean()
                if epoch_i == epochs-1:
                    # Save perfs / nb epoch 
                    stuff = precision_recall_fscore_support(true, pred)

        print("")
        print("Training complete!")
        self.model = model
        torch.cuda.empty_cache() 
        return stuff, best_perfs, best_epoch

    def save(self, saved_model_path):
         # SAVE
        output_dir = saved_model_path
        try:
            os.makedirs(output_dir)
        except:
            pass
        model = self.model

        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #tokenizer.save_vocabulary(output_dir)

    def load(self, saved_model_path):
         # LOAD
        if self.bert_model == "CamemBert":
            self.model = CamembertForSequenceClassification.from_pretrained(saved_model_path)
            self.model.cuda()
        elif self.bert_model == "Bert":
            self.model = BertForSequenceClassification.from_pretrained(saved_model_path)
            self.model.cuda()
        else:
            raise NotImplementedError

    def predict(self, inputs):
        model = self.model
        batch_size = self.modelargs['batch_size']

        input_ids, mask = inputs
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        # for our model.
        validation_inputs = torch.tensor(input_ids)
        validation_masks = torch.tensor(mask)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        model.eval()
        # acc_trunc_sent = ''
        # nb_O = 0
        # nb_OFF = 0
        # occurences = 0
        comp_labels = []
        comp_logits = []
        device = torch.device("cuda")
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            # Move logits and labels to CPU


            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            lab = np.argmax(logits, axis=1).flatten()
            logs = np.max(logits, axis=1).flatten()
            comp_labels = np.concatenate([comp_labels, lab])
            comp_logits = np.concatenate([comp_logits, logs])
        torch.cuda.empty_cache() 
        return comp_labels, comp_logits



class BertSequence():
    def __init__(self, config=None, bert_model=None):
        """
        Requires either a config, or manually-passed arguments following the data config template
        """
        if config is not None:
            self.data = config.data
            self.model = config.model
            self.train = config.train

        if bert_model is not None:
            self.bert_model = bert_model
        else:
            self.bert_model = "CamemBert"

    def fit_evaluate(self, train, test, **kwargs):
        if not hasattr(self, 'model') or self.model=={}:
            self.modelargs = kwargs
        
        # Check for params presence & define them
        batch_size = self.modelargs['batch_size']
        sampler = self.modelargs['sampler']
        epochs = self.modelargs['nepochs']

        if 'only_eval_last' in self.modelargs.keys() and self.modelargs['only_eval_last'] == True:
            only_eval_last = True
        else:
            only_eval_last = False

        try:
            seed_val = self.modelargs['random_seed']
        except:
            seed_val = 42
        try:
            lr = self.modelargs['learning_rate']
        except:
            lr = 5e-5
        try:
            eps = self.modelargs['eps']
        except:
            eps = 1e-8


        # Decompose train and test as subvectors
        train_inputs, train_labels, train_masks = train
        validation_inputs, validation_labels, validation_masks = test

        # Put train & test in tensors
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        # Need for nb of labels. numpy unique automatically flattens when needed
        nb_labels = len(np.unique(np.concatenate([train_labels, validation_labels])))

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
        nb_gpu = torch.cuda.device_count()

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        if sampler=='sequential':
            train_sampler = SequentialSampler(train_data)
        elif sampler=='random':
            train_sampler = RandomSampler(train_data)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_data, 
                                    sampler=train_sampler,
                                    batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(
            validation_data,
            sampler=validation_sampler,
            batch_size=batch_size)
        # Model: 
        if self.bert_model == "CamemBert":
            model = CamembertForTokenClassification.from_pretrained(
                    "camembert-base",
                    num_labels = nb_labels, 
                    output_attentions = False,
                    output_hidden_states = False,
                )
        elif self.bert_model == "Bert":
            model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels = nb_labels, 
                output_attentions = False,
                output_hidden_states = False,
                )
        else:
            raise NotImplementedError
        # TODO : pass cuda and device AS A PARAMETER
        device = torch.device("cuda")
        model.cuda()
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                        lr = lr,
                        eps = eps 
                        )

        # Number of training epochs (authors recommend between 2 and 4)
        #epochs = 3

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []
        # For each epoch...
        best_score = 0

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            t0 = time.time()
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            # Reset the total loss for this epoch.
            total_loss = 0

            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()

                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels) 
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            if only_eval_last == False or epoch_i == epochs-1:

                print("")
                print("Running Validation...")

                t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                model.eval()

                # Tracking variables 
                true = []
                pred = []
                # Evaluate data for one epoch
                for batch in validation_dataloader:
                # TODO : adapt evaluation to batch

                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)

                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch

                    # Telling the model not to compute or store gradients, saving memory and
                    # speeding up validation
                    with torch.no_grad():        

                        # Forward pass, calculate logit predictions.
                        # This will return the logits rather than the loss because we have
                        # not provided labels.
                        # token_type_ids is the same as the "segment ids", which 
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        # The documentation for this `model` function is here: 
                        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                        outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask)

                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    logits = outputs[0]

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    mask = b_input_mask.detach().cpu()

                    #print(np.argmax(logits, axis=2))
                    #print(label_ids.shape)
                    lab = np.multiply(np.argmax(logits, axis=2), mask).reshape([-1])
                    true_lab = np.multiply(label_ids, mask).reshape([-1])

                    true = np.concatenate([true, true_lab])
                    pred = np.concatenate([pred, lab])

                #print(classification_report(true, pred))
                perfs  = precision_recall_fscore_support(true, pred)
                if perfs[2].mean() > best_score:
                    best_perfs = perfs
                    best_epoch = epoch_i
                    best_score = perfs[2].mean()
                if epoch_i == epochs-1:
                    # Save perfs / nb epoch 
                    stuff = precision_recall_fscore_support(true, pred)
                # Report the final accuracy for this validation run.
                #print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
                print(classification_report(true, pred))
                # print("  Validation took: {:}".format(format_time(time.time() - t0)))
                
                
                if epoch_i == epochs-1:
                    # Save perfs / nb epoch 
                    stuff = precision_recall_fscore_support(true, pred)
                    # TODO : rename


        print("")
        print("Training complete!")
        self.model = model
        torch.cuda.empty_cache() 
        return stuff, best_perfs, best_epoch

    def save(self, saved_model_path):
         # SAVE
        output_dir = saved_model_path
        try:
            os.makedirs(output_dir)
        except:
            pass
        model = self.model

        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #tokenizer.save_vocabulary(output_dir)

    def load(self, saved_model_path):
         # LOAD
        if self.bert_model == "CamemBert":
            self.model = CamembertForTokenClassification.from_pretrained(saved_model_path)
            self.model.cuda()
        elif self.bert_model == "Bert":
            self.model = BertForTokenClassification.from_pretrained(saved_model_path)
            self.model.cuda()
        else:
            raise NotImplementedError

    ## predict 
    def predict(self, inputs):
        model = self.model
        batch_size = self.modelargs['batch_size']

        input_ids, mask = inputs
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        # for our model.
        validation_inputs = torch.tensor(input_ids)
        validation_masks = torch.tensor(mask)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        model.eval()
        # acc_trunc_sent = ''
        # nb_O = 0
        # nb_OFF = 0
        # occurences = 0
        truncated_labels = []
        truncated_logits = []
        device = torch.device("cuda")
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        


                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)


            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            m = b_input_mask.detach().cpu().numpy()
            #print(np.argmax(logits, axis=2))
            #print(label_ids.shape)
            lab = np.multiply(np.argmax(logits, axis=2), m)
            logs = np.multiply(np.max(logits, axis=2), m)
            sequence_length = np.sum(m, axis=1)
            i=0
            # inp = b_input_ids.detach().cpu().numpy()
            for l, lo in zip(lab, logs):
                predicted_labels = l[:sequence_length[i]]
                predicted_labels_prob = lo[:sequence_length[i]]
                truncated_labels.append(predicted_labels)
                truncated_logits.append(predicted_labels_prob)
                i=i+1
            #     tokens = self.tokenizer.convert_ids_to_tokens(inp[i][:sequence_length[i]])
            #     last_label = 0
            #     c = Counter(predicted_labels)
            #     nb_O = nb_O + c[1]
            #     nb_OFF = nb_OFF + c[2]
            #     for j in range(len(predicted_labels)):
            #         if predicted_labels[j]==2:
            #             tokens[j] = tokens[j] + " [OFF] "
            #             if last_label!= 2:
            #                 occurences = occurences + 1
            #         last_label = predicted_labels[j]
            #     tokens = self.tokenizer.convert_tokens_to_string(tokens)
            #     truncated_sequences.append(tokens)
            #     i = i+1
            # acc_trunc_sent = acc_trunc_sent + ' '.join(truncated_sequences)
                
        torch.cuda.empty_cache()        
        #return acc_trunc_sent, occurences, nb_OFF/nb_O
        return truncated_labels, truncated_logits


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def tokens2sents(tokens):
    sents = []
    last_token = ''
    acc_token = []
    for i in range(len(tokens)):
        acc_token.append(tokens[i])
        if tokens[i] in ['.', '__.', '?', '__?', '!', '__!']:
            if last_token in ['M', '▁M', 'Mr', '▁Mr', 'Mme', '▁Mme', 'm', '▁m', 'mr', '▁mr', '.', '▁.']:
                pass
            else:
                sents.append(acc_token)
                acc_token = []
        last_token = tokens[i]
    return sents