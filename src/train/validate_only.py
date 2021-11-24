import time
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification
import torch
import torchmetrics
from src.train.iterator_data_loader import load_data


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def validate_model(path:dict, validation_dataloader):
    t0 = time.time()
    global_val_step = 0
    
    today = datetime.date.today()
    date = today.strftime("%d.%m")
    save_path = os.path.join(os.getcwd(), "saved_models", date + "_validation")

    writer = SummaryWriter(save_path)

    model = BertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels = 9)

    for epoch_i in range(len(path)):

        if torch.cuda.is_available():    
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.load_state_dict(torch.load(path[epoch_i], map_location=device))
        model.to(device)
        model.eval()

        total_eval_loss = 0
        total_eval_tm_accuracy = 0
        total_eval_tm_precision = 0

        # Evaluate data for one epoch
        for batch_id, batch in enumerate(validation_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_punctuation_ids = batch[2].to(device)
            
            with torch.no_grad():        
                model_out = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_punctuation_ids)
                
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
            if batch_id % 1 == 0 and not batch_id == 0:
                writer.add_scalar("Validation loss", total_eval_loss / batch_id, global_step = global_val_step)
                writer.add_scalar("Torchmetrics validation accuracy", total_eval_tm_accuracy / batch_id, global_step = global_val_step)
                writer.add_scalar("Torchmetrics validation precision", total_eval_tm_precision / batch_id, global_step = global_val_step)

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

        
        #reset torchmetrics
        acc.reset()
        prec.reset()
        


path1 = os.path.join("saved_models", "23.11", "epoch1_model.pt")
path2 = os.path.join("saved_models", "23.11", "epoch2_model.pt")
path3 = os.path.join("saved_models", "23.11", "epoch3_model.pt")
path4 = os.path.join("saved_models", "23.11", "epoch4_model.pt")
path = {0: path1, 1: path2, 2: path3, 3: path4}

train_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "training_data.pt")
val_path = os.path.join(os.getcwd(), "data", "processed", "dereko", "tensors", "datasets", "validation_data.pt")
train_data = torch.load(train_path)
val_data = torch.load(val_path)
train_dataloader, validation_dataloader = load_data(train_data, val_data)

validate_model(path, validation_dataloader)