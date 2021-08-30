from torch import tensor
from torch.utils.data import TensorDataset, random_split
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
import torch



def split_train_validate(input_ids, attention_masks, punctuation_ids : torch.Tensor) -> TensorDataset:
    """Creates a train-validation split.
    Args: 
        input_ids: Tensors with tokenized sentences with special tokens and padding.
        punctuation_ids: Corresponding ID of the punctuation of a specific spot in a sentence.
        attention_masks:  Corresponding attention mask.
    Returns: 
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation
    """

    # Combine the training inputs into a TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, punctuation_ids)

    # Create a 80-20 train-validation split.

    # Calculate the number of samples to include in each set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset


load_path = os.path.join(PROCESSED_DATA_PATH, "tensors")
input_ids = torch.load(os.path.join(load_path, "input_ids.pt"))
attention_masks = torch.load(os.path.join(load_path, "attention_masks.pt"))
punctuation_ids = torch.load(os.path.join(load_path, "punctuation_ids.pt"))
train_dataset, val_dataset = split_train_validate(input_ids, attention_masks, punctuation_ids)
torch.save(train_dataset, os.path.join(load_path, "datasets", "training_data.pt"))
torch.save(val_dataset, os.path.join(load_path, "datasets", "validation_data.pt"))