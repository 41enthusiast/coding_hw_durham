from sklearn.model_selection import train_test_split

def get_train_val_split(data_dir, get_n_classes):

    data_dir = ds_name

    train_split = int(0.8 * len(dataset))
    val_split = len(dataset) - int(0.9 * len(dataset))
    test_split = len(dataset) - train_split - val_split

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        shuffle=True,
        stratify=y
    )
    return train_idx, val_idx