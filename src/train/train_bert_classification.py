import numpy as np
import time
import datetime
import random
from numpy.lib.function_base import average
from torchmetrics.classification.accuracy import Accuracy
#from torchmetrics.text import bert
from transformers import BertForSequenceClassification, AdamW
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

# Function to calculate the accuracy of our predictions vs punctuation_ids
def flat_accuracy(preds, punctuation_ids):
    pred_flat = np.argmax(preds.detach().cpu().numpy(), axis=1).flatten()
    punctuation_ids_flat = punctuation_ids.flatten()
    return np.sum(pred_flat == punctuation_ids_flat) / len(punctuation_ids_flat)



def trainBertClassification(train_dataloader, validation_dataloader):

    # Load BertForTokenClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    #model = BertForTokenClassification.from_pretrained('bert-base-german-cased')
    
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-german-cased", # Use the German BERT model, with an cased vocab. More information here: https://www.deepset.ai/german-bert
    num_labels = 9, # The number of output punctuation_ids--9, multi-class task.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    ) 

    #model = BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), "saved_models", "trained_model_03_09.pt"))
    
    #print(model.parameters)
    # If there's a GPU available...
    if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Tell pytorch to run this model on the GPU.
    #model.cuda()
    # Tell pytorch to run this model on the available device.
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, BERT authors recommend: 5e-5, 3e-5, 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # specify a folder for the TensorBoard-writer
    writer = SummaryWriter('src/train/training_logs/')

    #initialize torchmetrics
    acc = torchmetrics.Accuracy(num_classes=9, average="macro")
    acc.to(device)
    prec = torchmetrics.Precision(num_classes=9, average="macro")
    prec.to(device)
    f1 = torchmetrics.F1(num_classes=9, average="micro")
    f1.to(device)

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
                model.eval()
                torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_models", "epoch{:}_model.pt".format(epoch_i + 1)))
                model.train()

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the CPU or GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: punctuation ids 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_punctuation_ids = batch[2].to(device)
            #b_punctuation_ids = batch[2].resize_(batch[2].size(0),b_input_ids.size(1)) #punct_ids had to resized to be the same size as other tensors
            #b_punctuation_ids = b_punctuation_ids.to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            #print(step)
            model_out = model(input_ids = b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_punctuation_ids)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += model_out.loss.item()

            # Perform a backward pass to calculate the gradients.
            model_out.loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            #log training loss
            writer.add_scalar("Training loss", model_out.loss.item(), global_step = step)
            
            #log training accuracy
            accuracy = acc(model_out.logits, b_punctuation_ids)
            accuracy = acc.compute()
            precision = prec(model_out.logits, b_punctuation_ids)
            precision = prec.compute()
            f1_score = f1(model_out.logits, b_punctuation_ids)

            writer.add_scalar("Torchmetrics accuracy", accuracy, global_step = step)
            writer.add_scalar("Torchmetrics precision", precision, global_step = step)
            


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)        
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoсh took: {:}".format(training_time))
            
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

        # Save the current state
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_models", "trained_model.pt"))
        model.save_pretrained(os.path.join(os.getcwd(), "saved_models"))

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the CPU or GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: punctuation_ids 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_punctuation_ids = batch[2].to(device)
            
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
                model_out = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_punctuation_ids)
                
            # Accumulate the validation loss.
            total_eval_loss += model_out.loss.item()


            # Move logits and punctuation_ids to CPU
            logits = model_out.logits.detach().cpu().numpy()
            punctuation_ids = b_punctuation_ids.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(model_out.logits, punctuation_ids)

            #log validation loss
            writer.add_scalar("Validation loss", model_out.loss.item(), global_step = step)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        #compute torchmetrics-accuracy
        accuracy = acc.compute()
        precision = prec.compute()
        f1_score = f1.compute()

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'torchmetrics Accuracy': str(accuracy),
                'torchmetrics AP': str(precision),
                'torchmetrics F1': str(f1_score)
            }
        )
        #reset torchmetrics
        acc.reset()
        prec.reset()
        f1.reset()
        
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        #creating a dummy input, for the TB graph
        #dummy_input = torch.randint(1, 9, (32, 40)) #low, high, size(tuple)
        #wrapping model in another class that converts outputs from dict into namedtuple for graph visualization
        #model_wrapper = ModelWrapper(model)
        #writer.add_graph(model_wrapper, dummy_input)
        save_to_json(training_stats, "src/train/training_logs/manual_log.json")

def main():
    train_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "test_training_data.pt")
    val_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "test_validation_data.pt")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    train_dataloader, validation_dataloader = load_data(train_data, val_data)
    trainBertClassification(train_dataloader, validation_dataloader)

if __name__ == "__main__":
    main()