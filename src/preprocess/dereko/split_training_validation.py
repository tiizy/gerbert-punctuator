from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
from torch.utils.data import TensorDataset, random_split


def split_train_validate(input_ids, attention_masks, punctuation_ids) -> TensorDataset:
    """Creates a train-validation split.
    Args: 
        input_ids: Tensors with tokenized sentences with special tokens and padding.
        punctuation_ids: Corresponding ID of the punctuation of a specific spot in a sentence.
        attention_masks:  Corresponding attention mask.
    Returns: 
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation
    """

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, punctuation_ids)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset