import numpy as np
import time
import datetime
import random
from numpy.lib.function_base import average
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.tensorboard import SummaryWriter #SummaryWriter: key element to TensorBoard
from src.preprocess.utils.json_handler import save_to_json
#from src.train.model_wrapper import ModelWrapper
import torchmetrics

import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.train.iterator_data_loader import load_data


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def trainBertClassification(train_dataloader, validation_dataloader):

 
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-german-cased", # Use the German BERT model, with an cased vocab. More information here: https://www.deepset.ai/german-bert
    num_labels = 9, # The number of output punctuation_ids--9, multi-class task.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    #adding special tokens to a tokenizer and resizing the model accordingly
    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased", do_lower_case = False)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<PUNCT>']})
    model.resize_token_embeddings(len(tokenizer))

    #print(model.parameters)
    # If there's a GPU available...
    if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Tell pytorch to run this model on the available device.
    model.to(device)

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, BERT authors recommend: 5e-5, 3e-5, 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    # Number of training epochs.
    epochs = 4

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    # Set the seed value to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # Add a save path with a current date
    today = datetime.date.today()
    date = today.strftime("%d.%m")
    save_path = os.path.join(os.getcwd(), "saved_models", date)

    # specify a folder for the TensorBoard-writer
    writer = SummaryWriter(save_path)

    #initialize torchmetrics
    acc = torchmetrics.Accuracy(num_classes=9, average="macro")
    acc_class = torchmetrics.Accuracy(num_classes=9, average="none")
    acc.to(device)
    acc_class.to(device)
    prec = torchmetrics.Precision(num_classes=9, average="macro")
    prec_class = torchmetrics.Precision(num_classes=9, average="none")
    prec.to(device)
    prec_class.to(device)
    #Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    f1 = torchmetrics.F1(num_classes=9, average="macro")
    f1_class = torchmetrics.F1(num_classes=9, average="none")
    f1.to(device)
    f1_class.to(device)

    counter_t = 0
    global_val_step = 0

    for epoch_i in range(0, epochs):

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_tm_accuracy = 0
        total_tm_accuracy_class = 0
        total_tm_precision = 0
        total_tm_precision_class = 0
        total_tm_f1 = 0
        total_tm_f1_class = 0

        total_train_loss_t = 0
        total_tm_accuracy_t = 0
        total_tm_precision_t = 0
        total_tm_f1_t = 0
        
        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        print("Total steps in one epoch: " + str(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            #Save model every 1000 steps
            if step % 1000 == 0 and not step == 0:
                counter_t += 1

                model.eval()
                torch.save(model.state_dict(), os.path.join(save_path, "epoch{:}_model.pt".format(epoch_i + 1)))
                
                avg_train_loss = total_train_loss_t / 1000
                avg_tm_accuracy = total_tm_accuracy_t / 1000
                avg_tm_precision = total_tm_precision_t / 1000
                avg_tm_f1 = total_tm_f1_t / 1000
                
                writer.add_scalar("Average Training loss per 1000 steps", avg_train_loss, counter_t)
                writer.add_scalar("Average Torchmetrics accuracy per 1000 steps", avg_tm_accuracy, counter_t)
                writer.add_scalar("Average Torchmetrics precision per 1000 steps", avg_tm_precision, counter_t)
                writer.add_scalar("Average Torchmetrics f1 per 1000 steps", avg_tm_f1, counter_t)

                total_train_loss_t = 0
                total_tm_accuracy_t = 0
                total_tm_precision_t = 0
                total_tm_f1_t = 0
                
                model.train()

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_punctuation_ids = batch[2].to(device)

            #clearing any previously calculated gradients before performing a backward pass
            model.zero_grad()        

            #performing a forward pass
            model_out = model(input_ids = b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_punctuation_ids)

            #accumulating the training loss over all of the batches so that we can calculate the average loss
            total_train_loss += model_out.loss.item()
            total_train_loss_t += model_out.loss.item()

            #performing a backward pass to calculate the gradients
            model_out.loss.backward()

            #clipping the norm of the gradients to 1.0, to prevent the "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #log training accuracy, precision, f1
            accuracy = acc(model_out.logits, b_punctuation_ids)
            accuracy_class = acc_class(model_out.logits, b_punctuation_ids)
            precision = prec(model_out.logits, b_punctuation_ids)
            precision_class = prec_class(model_out.logits, b_punctuation_ids)
            f1_score = f1(model_out.logits, b_punctuation_ids)
            f1_for_class = f1_class(model_out.logits, b_punctuation_ids)

            total_tm_accuracy += accuracy
            total_tm_accuracy_t += accuracy
            total_tm_accuracy_class += accuracy_class
            total_tm_precision += precision
            total_tm_precision_t += precision
            total_tm_precision_class += precision_class
            total_tm_f1 += f1_score
            total_tm_f1_t += f1_score
            total_tm_f1_class += f1_for_class

            #updating parameters and take a step using the computed gradient
            optimizer.step()

            #updating the learning rate
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_tm_accuracy = total_tm_accuracy / len(train_dataloader)
        avg_tm_accuracy_class = total_tm_accuracy_class / len(train_dataloader)
        avg_tm_precision = total_tm_precision / len(train_dataloader)
        avg_tm_precision_class = total_tm_precision_class / len(train_dataloader)
        avg_tm_f1 = total_tm_f1 / len(train_dataloader)
        avg_tm_f1_class = total_tm_f1_class / len(train_dataloader)

        writer.add_scalar("Average Training loss per epoch", avg_train_loss, epoch_i)
        writer.add_scalar("Average Torchmetrics accuracy per epoch", avg_tm_accuracy, epoch_i)
        writer.add_scalar("Average Torchmetrics precision per epoch", avg_tm_precision, epoch_i)
        writer.add_scalar("Average Torchmetrics f1 per epoch", avg_tm_f1, epoch_i)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        #reset torchmetrics
        acc.reset()
        acc_class.reset()
        prec.reset()
        prec_class.reset()
        f1.reset()
        f1_class.reset()
            

        #measuring our performance on the validation set

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Save the current state
        torch.save(model.state_dict(), os.path.join(save_path, "trained_model.pt"))
        model.save_pretrained(save_path)

        # Tracking variables 
        total_eval_loss = 0
        total_eval_tm_accuracy = 0
        total_eval_tm_precision = 0

        for batch_id, batch in enumerate(validation_dataloader):
            
            #unpacking this training batch from the dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_punctuation_ids = batch[2].to(device)
            
            #preventing pytorch from creating the compute graph during the forward pass
            with torch.no_grad():        

                model_out = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_punctuation_ids)
                
            # Accumulate the validation loss.
            total_eval_loss += model_out.loss.item()


            #initialize torchmetrics
            acc = torchmetrics.Accuracy(num_classes=9, average="macro")
            acc.to(device)
            prec = torchmetrics.Precision(num_classes=9, average="macro")
            prec.to(device)

            #log validation loss, accuracy, precision, f1 
            accuracy = acc(model_out.logits, b_punctuation_ids)
            precision = prec(model_out.logits, b_punctuation_ids)
            
            total_eval_tm_accuracy += accuracy
            total_eval_tm_precision += precision

            
            global_val_step += 1
            if batch_id % 100 == 0 and not batch_id == 0:
                writer.add_scalar("Validation loss", total_eval_loss / batch_id, global_step = global_val_step)
                writer.add_scalar("Torchmetrics validation accuracy", total_eval_tm_accuracy / batch_id, global_step = global_val_step)
                writer.add_scalar("Torchmetrics validation precision", total_eval_tm_precision / batch_id, global_step = global_val_step)

        #calculating the average metrics over all of the batches
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        avg_val_tm_acc = total_eval_tm_accuracy / len(validation_dataloader)
        avg_val_tm_prec = total_eval_tm_precision / len(validation_dataloader)

        writer.add_scalar("Average Validation loss per epoch", avg_val_loss, global_step = epoch_i)
        writer.add_scalar("Average Validation accuracy per epoch", avg_val_tm_acc, global_step = epoch_i)
        writer.add_scalar("Average Validation precisionper epoch", avg_val_tm_prec, global_step = epoch_i)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'torchmetrics Accuracy': str(avg_tm_accuracy),
                'torchmetrics Accuracy per class': str(avg_tm_accuracy_class),
                'torchmetrics Precision': str(avg_tm_precision),
                'torchmetrics Precision per class': str(avg_tm_precision_class),
                'torchmetrics F1': str(avg_tm_f1),
                'torchmetrics F1 per class': str(avg_tm_f1_class),
                'tochmetrics valid. Accuracy': str(avg_val_tm_acc),
                'tochmetrics valid. Precision': str(avg_val_tm_prec)
            }
        )
        #reset torchmetrics
        acc.reset()
        prec.reset()
        f1.reset()
        
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        save_to_json(training_stats, os.path.join(save_path, "manual_log.json"))

def main():
    train_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "training_data.pt")
    val_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "validation_data.pt")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    train_dataloader, validation_dataloader = load_data(train_data, val_data)
    trainBertClassification(train_dataloader, validation_dataloader)

if __name__ == "__main__":
    main()