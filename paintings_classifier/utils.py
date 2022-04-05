from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import torch
import itertools
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# evaluating the model
def evaluate(data_loader, module):
    with torch.no_grad():
        progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
        module.eval().cuda()
        true_y, pred_y = [], []
        for i, batch_ in enumerate(data_loader):
            X, y = batch_
            print(progress[i % len(progress)], end="\r")
            y_pred = (module(X.cuda()).sigmoid() > 0.5).int().squeeze()#threshold of 0.5 chosen
            true_y.extend(y.squeeze().cpu())
            pred_y.extend(y_pred.cpu())
        print(classification_report(true_y, pred_y, digits=3))
        return true_y, pred_y


def get_train_val_split(args, get_n_classes):
    # set up the dataset
    img_size = (32, 32)  # placeholder image size
    temp_transforms = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor()])  # , transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.ds_name, transform=temp_transforms)
    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=len(dataset),
        # collate_fn=training.collate_pil
    )
    x, y = next(iter(loader))

    # get the number of classes
    if get_n_classes:
        n_classes = len(dataset.classes)
    else:
        n_classes = None

    # train val test split from the total dataset
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        shuffle=True,
        stratify=y
    )
    return train_idx, val_idx, n_classes


def make_confusion_matrix(module, loader, device):
    @torch.no_grad()
    def get_all_preds(model):
        all_preds = torch.tensor([]).to(device)
        all_tgts = torch.tensor([]).to(device)
        for batch in loader:
            images, labels = batch
            preds = model(images.to(device))
            all_preds = torch.cat(
                (all_preds, preds)
                , dim=0
            )
            all_tgts = torch.cat(
                (all_tgts, labels.to(device))
                , dim=0
            )
        return all_preds, all_tgts

    # set up model predictions and targets in right format for making the confusion matrix
    preds, tgts = get_all_preds(module.model.to(device), module.val_dataloader(), device)
    stacked = torch.stack(
        (
            tgts.squeeze()
            , (preds.sigmoid() > 0.5).int().squeeze()  # threshold of 0.5 chosen
        )
        , dim=1
    )

    # make the confusion matrix
    cm = torch.zeros(module.hparam.n_classes, module.hparam.n_classes, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cm[int(tl), int(pl)] = cm[int(tl), int(pl)] + 1

    return cm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # set up the confusion matrix visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_most_and_least_confident_predictions(module, device):
    # get model prediction for the validation dataset and all images for later use
    preds = torch.tensor([]).to(device)
    all_images = torch.tensor([]).to(device)
    loader = module.val_dataloader()
    model = module.model.to(device)
    for batch in loader:
        images, _ = batch
        pred_batch = model(images.to(device))
        preds = torch.cat(
            (preds, pred_batch)
            , dim=0
        )
        all_images = torch.cat(
            (all_images, images.to(device)),
            dim=0
        )
    confidence = preds.sigmoid()

    # get indices with most and least confident scores
    most_confident = confidence.topk(4, dim=0).indices
    least_confident = confidence.topk(4, dim=0, largest=False).indices

    # get the images according to confidence scores, 4 each
    mc_imgs = make_grid(all_images[most_confident.squeeze()], 4)
    lc_imgs = make_grid(all_images[least_confident.squeeze()], 4)

    return mc_imgs, lc_imgs