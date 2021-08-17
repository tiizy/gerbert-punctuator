from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_data(train_dataset, val_dataset) -> DataLoader:
    """Effectively iterates through datasets.
    Args: 
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation
    Returns: 
        train_dataloader: An iteration for training data.
        validation_dataloader = An iteration for validation data.
    """

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader, validation_dataloader
