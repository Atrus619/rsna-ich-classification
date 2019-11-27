import os
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib


class ResNeXt(nn.Module):
    def __init__(self, num_classes, device):
        super(ResNeXt, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.resnext = models.resnext101_32x8d(pretrained=True).to(self.device)
        self.resnext.fc = nn.Linear(in_features=self.resnext.fc.in_features, out_features=num_classes).to(self.device)

        self.features = nn.Sequential(
            self.resnext.conv1,
            self.resnext.bn1,
            self.resnext.relu,
            self.resnext.maxpool,
            self.resnext.layer1,
            self.resnext.layer2,
            self.resnext.layer3,
            self.resnext.layer4
        )

        self.avgpool = self.resnext.avgpool
        self.classifier = self.resnext.fc

        self.gradients = None

    def forward(self, x):
        # Extract features
        x = self.features(x)

        # Stop at final conv output and register hook for Grad CAM
        h = x.register_hook(self.activations_hook)

        # Continue
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    @torch.utils.hooks.unserializable_hook
    def activations_hook(self, grad):
        """
        Used for Grad CAM
        Hook for the gradients of the activations
        Used on the final convolutional layer
        """
        self.gradients = grad

    def get_activations_gradient(self):
        """Grad CAM Helper Function"""
        return self.gradients

    def get_activations(self, x):
        """Grad CAM Helper Function"""
        return self.features(x)

    def draw_cam(self, img, true_label=None, scale=255, path='tmp.jpg', show=True):
        """
        Implements Grad CAM
        :param img: Image to draw over
        :param true_label: True label for displaying in image. If none, will be ignored.
        :param scale: Multiplier to scale heatmap by
        :param path: Path to save output image to. Full image path that should end in .jpg
        :param show: Whether to show the image
        :return: Set of four images, one showing the original, and the remainder showing the activation for each class
        """
        self.eval()

        # Preprocess inputs
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        # For each label
        for j in range(self.num_classes):
            pred = self.forward(img).squeeze()[j]
            pred.backward()
            gradients = self.get_activations_gradient()
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations = self.get_activations(img).detach()

            # Weight the channels by corresponding gradients
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]

            # Average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()

            # ReLU on top of the heatmap (possibly should use the actual activation in the network???)
            heatmap = np.maximum(heatmap, 0)

            # Normalize heatmap
            heatmap /= torch.max(heatmap)
            heatmap = heatmap.numpy()

            # Save original image
            img_transformed = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            matplotlib.image.imsave(path, img_transformed, cmap=plt.cm.bone)

            # Read in image and cut pixels in half for visibility
            cv_img = cv2.imread(path)
            cv_img = cv_img / 2

            # Create heatmap
            heatmap = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
            heatmap = np.uint8(scale * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose
            superimposed_img = heatmap * 0.4 + cv_img

            # Save
            label_specific_path = (' ' + str(j) + '.').join(path.split('.'))
            cv2.imwrite(label_specific_path, superimposed_img)

        # Load in and make pretty
        imgs = []
        m = nn.Sigmoid()
        confidences = m(self.forward(img).squeeze())
        for i in range(self.num_classes):
            label_specific_path = (' ' + str(i) + '.').join(path.split('.'))
            imgs.append(plt.imread(label_specific_path))
            os.remove(label_specific_path)

        f, axes = plt.subplots(2, 2, figsize=(16, 16))
        plt.sca(axes[0, 0])
        plt.axis('off')
        plt.title('Original Image', fontweight='bold')
        img = img.squeeze().detach().cpu().numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap=plt.cm.bone)

        add_str = ''  # Used if true_label is passed, otherwise empty string

        plt.sca(axes[0, 1])
        plt.axis('off')
        if true_label is not None:
            add_str = ' (Present)' if true_label[0] else ' (Not Present)'
        plt.title(f'Grad CAM for Epidural w/ confidence {confidences[0]:.2%}{add_str}', fontweight='bold')
        plt.imshow(imgs[0])

        plt.sca(axes[1, 0])
        plt.axis('off')
        if true_label is not None:
            add_str = ' (Present)' if true_label[1] else ' (Not Present)'
        plt.title(f'Grad CAM for Intraparenchymal w/ confidence {confidences[1]:.2%}{add_str}', fontweight='bold')
        plt.imshow(imgs[1])

        plt.sca(axes[1, 1])
        plt.axis('off')
        if true_label is not None:
            add_str = ' (Present)' if true_label[2] else ' (Not Present)'
        plt.title(f'Grad CAM for Subarachnoid w/ confidence {confidences[2]:.2%}{add_str}', fontweight='bold')
        plt.imshow(imgs[2])

        sup = 'Gradient Class Activation Map'
        st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        f.savefig(path)

        if show:
            plt.show()
