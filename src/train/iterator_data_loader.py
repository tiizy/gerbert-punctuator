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

    batch_size = 32

    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size
            )
    return train_dataloader, validation_dataloader
