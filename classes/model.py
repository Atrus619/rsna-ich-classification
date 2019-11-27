import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import copy
from tqdm import tqdm
from utils import get_pretty_time
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class FineTuneTrainer:
    def __init__(self, pretrained_model, device, num_classes, loss_fn, lr, lr_decay_freq=1, lr_decay=0.95, max_norm=1.0, fp16=None):
        self.device = device

        self.model = pretrained_model
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes).to(self.device)
        self.model = self.model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_freq, gamma=lr_decay)

        self.current_step = 0
        self.epoch = 1
        self.num_epochs = 0

        self.training_losses = []
        self.validation_losses = []

        self.max_norm = max_norm
        self.fp16 = fp16
        if self.fp16:
            from apex import amp  # Apex is only required if we use fp16 training
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16)

    def train_one_epoch(self, train_loader, grad_accum_steps):
        self.model.train()

        running_loss = 0.0
        running_count = 0

        with tqdm(train_loader, desc=f'Epoch: {self.epoch} Training') as t:
            for batch in t:
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                batch_loss = self.loss_fn(outputs, labels)

                # Apply gradients
                if self.fp16:
                    from apex import amp
                    with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_norm)
                else:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

                self.current_step += 1

                if self.current_step % grad_accum_steps == 0:
                    self.current_step = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Statistics
                running_loss += batch_loss.item() * images.size(0)
                running_count += images.size(0)
                t.set_postfix_str(f'Loss: {running_loss / running_count:.4f}')

        train_loss = running_loss / running_count
        self.next_epoch(train_loss=train_loss)

    def eval_once(self, val_loader):
        self.model.eval()

        running_loss = 0.0
        running_count = 0

        with tqdm(val_loader, desc=f'Epoch: {self.epoch} Validation') as t:
            for batch in t:
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)

                with torch.no_grad():
                    outputs = self.model(images)
                    batch_loss = self.loss_fn(outputs, labels)

                    running_loss += batch_loss.item() * images.size(0)
                    running_count += images.size(0)
                    t.set_postfix_str(f'Loss: {running_loss / running_count:.4f}')

        val_loss = running_loss / running_count
        self.next_epoch(val_loss=val_loss)

    def train_model(self, train_loader, val_loader, num_epochs, val_freq, grad_accum_steps, eval_first=False, early_stopping_patience=None, oversample=False):
        start_time = time.time()
        print(f'Beginning training for {num_epochs} epochs.')
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        self.num_epochs = num_epochs

        if eval_first:
            self.eval_once(val_loader=val_loader)

        for epoch in range(1, num_epochs + 1):
            if oversample:
                train_loader.dataset.reapply_oversample()
            self.train_one_epoch(train_loader=train_loader, grad_accum_steps=grad_accum_steps)

            if epoch % val_freq == 0:
                self.eval_once(val_loader=val_loader)
                if min(self.validation_losses) == self.validation_losses[-1]:
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    best_epoch = self.epoch

                if early_stopping_patience and len(self.validation_losses) >= early_stopping_patience:
                    if min(self.validation_losses) != min(self.validation_losses[-early_stopping_patience:]):
                        print(f'Early stopping with patience {early_stopping_patience} triggered at epoch {epoch}.')
                        break

            self.scheduler.step()

        print(f'Training complete. Total training time: {get_pretty_time(time.time() - start_time)}')

        print(f'Best model occurred at epoch {best_epoch}. Reloading weights from this checkpoint.')
        self.model.load_state_dict(best_model_weights)

    def next_epoch(self, train_loss=None, val_loss=None):
        if train_loss:
            self.epoch += 1
            self.training_losses.append(train_loss)

        if val_loss:
            self.validation_losses.append(val_loss)

    def freeze_first_n_trainable_layers(self, trainable_layers, n=None, freeze=True):
        """
        Uses recursion to access all layers in model. Three modes:
            1. n=None: Print/return a list of the layers based on trainable_layers that can be used to determine the point at which to start freezing/unfreezing.
            2. n=Num and freeze=True: Freeze all layers up through the layer specified (based on the index in the list returned from mode #1.
            3. n=Num and freeze=False: Unfreeze ""
        :param trainable_layers: Set containing strings of layer names to target
        :param n: Number of layers through which to freeze/unfreeze starting from the beginning
        :param freeze: Boolean as to whether to freeze or unfreeze
        """
        OUTPUT = []

        def recursive_network_traversal(node, trainable_layers, freeze, parent_name='', freeze_first_n=None):
            """
            Recursive function to traverse nested layers. Relies on a global variable OUTPUT existing (init as []).
            Two modes exist:
                1. Freeze_first_n is None will result in a list of the flattened network being created and printed out.
                2. Freeze_first_n is an index in the list up to which all parameters should have requires_grad set to False.
            """
            if len(list(node.children())) == 0:  # Reached bottom of tree
                if node._get_name() in trainable_layers:
                    full_layer_desc = parent_name + ' - ' + node._get_name()
                    if freeze_first_n:
                        if len(OUTPUT) <= freeze_first_n:
                            print('Freezing parameters for layer: ' + full_layer_desc)
                        else:
                            print('Not freezing parameters for layer: ' + full_layer_desc)
                    else:
                        print(str(len(OUTPUT)) + '. ' + full_layer_desc)

                    OUTPUT.append(full_layer_desc)

                if freeze_first_n and (len(OUTPUT) - 1) <= freeze_first_n:  # Freeze all parameters as long as we are currently freezing them.
                    for param in node.parameters():
                        param.requires_grad = freeze

            else:
                for child in node.children():
                    full_name = (parent_name + ' - ' if parent_name != '' else '') + node._get_name()
                    recursive_network_traversal(node=child, trainable_layers=trainable_layers, freeze=freeze, parent_name=full_name, freeze_first_n=freeze_first_n)

        recursive_network_traversal(node=self.model, trainable_layers=trainable_layers, freeze=freeze, freeze_first_n=n)
        if n:
            return
        else:
            return OUTPUT

    def plot_learning_curve(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.training_losses, label='Train')
        ax.plot(self.validation_losses, label='Validation')

        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        ax.legend(loc='upper center', shadow=True, ncol=2)
        plt.show()

    def produce_predictions(self, test_loader, threshold=0.5, test=True):
        """
        Creates two dfs, one with probabilities and one with deterministic classifications based on specified threshold
        :param test_loader: Loader to produce images
        :param threshold: Threshold to determine whether a class is positively identified.
        :param test: Whether the images come from test or train. Only affects the name of the images.
        :return: Two dataframes, the first containing the probabilities associated with a particular class, and the second containing the deterministic decisions based on specified threshold.
        """
        proba_output_df = pd.DataFrame(columns=['ID', 'epidural', 'intraparenchymal', 'subarachnoid'])
        decision_output_df = pd.DataFrame(columns=proba_output_df.columns)
        add_str = 'test_' if test else 'train_'
        m = nn.Sigmoid()

        with tqdm(test_loader, desc=f'Producing Outputs') as t:
            for batch in t:
                images, IDs = batch['image'].to(self.device), batch['IDs']

                with torch.no_grad():
                    outputs = m(self.model(images))

                raw_proba_output_df = pd.DataFrame(outputs.cpu().numpy(), columns=['epidural', 'intraparenchymal', 'subarachnoid'])
                raw_decision_output_df = raw_proba_output_df.applymap(lambda x: 1 if x > threshold else 0)
                id_df = pd.DataFrame(IDs.numpy(), columns=['ID'])
                id_df['ID'] = add_str + id_df['ID'].apply(str)
                tmp_proba_output_df = pd.concat((id_df, raw_proba_output_df), axis=1)
                tmp_decision_output_df = pd.concat((id_df, raw_decision_output_df), axis=1)

                proba_output_df = proba_output_df.append(tmp_proba_output_df)
                decision_output_df = decision_output_df.append(tmp_decision_output_df)

        return proba_output_df, decision_output_df

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f)
