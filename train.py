import numpy as np
import os
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations import HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Resize, CenterCrop, Compose
from albumentations.pytorch import ToTensor
import utils
from classes.dataset import ImageDataset
from classes.model import FineTuneTrainer
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from classes.ResNeXt import ResNeXt

# Customizable inputs
MODEL_NAME = 'MODEL_NAME_GOES_HERE'
NUM_EPOCHS = 35
VAL_FREQ = 1  # Number of epochs after which to run a validation
LR = 2e-6
DEVICE = 'cuda:0'  # Set to my one gpu
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2  # Increases effective batch size by a multiplier of this value
NUM_WORKERS = 16
NUM_CLASSES = 3
CSV_FILE = 'data/wetransfer-1c7414/behold_coding_challenge_train.csv'
RANDOM_STATE = 0
EARLY_STOPPING_PATIENCE = 5

# Set up datasets and dataloaders
train_indices, val_indices = utils.train_val_split(csv_file=CSV_FILE, test_prop=0.25, random_state=RANDOM_STATE)

training_transform = Compose([
    Resize(256, 256),
    CenterCrop(224, 224),
    HorizontalFlip(),
    RandomBrightnessContrast(),
    ShiftScaleRotate(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # based on: https://pytorch.org/hub/pytorch_vision_resnext/
    ToTensor(),
])
training_data = ImageDataset(idxs=train_indices.values,
                             csv_file=CSV_FILE,
                             root_dir='data/wetransfer-1c7414/train_images',
                             transform=training_transform,
                             oversample=True)

train_loader = DataLoader(training_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

val_transform = Compose([
    Resize(256, 256),
    CenterCrop(224, 224),
    ToTensor(),
])
val_data = ImageDataset(idxs=val_indices.values,
                        csv_file=CSV_FILE,
                        root_dir='data/wetransfer-1c7414/train_images',
                        transform=val_transform)

val_loader = DataLoader(val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS)

# Initialize model and trainer
loss_fn = nn.BCEWithLogitsLoss()

pretrained_model = models.resnext101_32x8d(pretrained=True)
model_trainer = FineTuneTrainer(pretrained_model=pretrained_model, device=DEVICE, num_classes=NUM_CLASSES, loss_fn=loss_fn, lr=LR, fp16="O1")
model_trainer.freeze_first_n_trainable_layers(trainable_layers={'Conv2d', 'Linear'}, n=97)

# Train model
model_trainer.train_model(train_loader=train_loader,
                          val_loader=val_loader,
                          num_epochs=NUM_EPOCHS,
                          val_freq=VAL_FREQ,
                          grad_accum_steps=GRAD_ACCUM_STEPS,
                          eval_first=True,
                          early_stopping_patience=EARLY_STOPPING_PATIENCE,
                          oversample=True)

model_trainer.save(os.path.join('logs', 'models', MODEL_NAME))

val_probs, val_decs = model_trainer.produce_predictions(val_loader, test=False)
y_pred = val_decs.iloc[:, 1:].values.astype('int64')
y_true = val_data.labels.iloc[:, 1:].values
utils.print_model_metrics(y_pred, y_true)

test_data = ImageDataset(root_dir='data/wetransfer-1c7414/test_images',
                         transform=val_transform)
test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

test_prob, test_dec = model_trainer.produce_predictions(test_loader)

test_prob.to_csv('binary_probabilities.csv', index=False)
test_dec.to_csv('binary_predictions.csv', index=False)
