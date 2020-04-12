"""
visualize results for test image

Usage:
    python visualize.py [input folder] [output folder]

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage import io

import os
from sys import argv

from models import *
from evaluate import ImageDataset

input_dir = argv[1]
output_dir = argv[2]

batch_size = 400

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
weight_path = os.path.join('FER2013_VGG19', 'PrivateTest_model.t7')
color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


test_loader = DataLoader(
    ImageDataset(input_dir), 
    batch_size=batch_size // 10,  # 10 crop
    shuffle=False, num_workers=4
)

# -------------
# Visualization
# -------------

def is_color(img, threshold=1):
    """
    check if the mean absolute difference to channel mean > threshold
    """
    if img.ndim < 3 or img.shape[0] == 1:
        return False
    avg = np.mean(img, axis=2, keepdims=True)
    return np.mean(np.abs(img - avg)) > threshold

def visualization(score, y_pred, fname, emoj=False):
    raw_img = io.imread(os.path.join(input_dir, fname))

    plt.rcParams['figure.figsize'] = (9.5 + emoj * 4, 5.5)
    axes = plt.subplot(1, 2 + emoj, 1)

    if is_color(raw_img):
        plt.imshow(raw_img)
    else:
        plt.imshow(raw_img, cmap='gray')

    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 2 + emoj, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    
    for i in range(len(class_names)):
        plt.bar(ind[i], score[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    if emoj:
        axes = plt.subplot(1, 2 + emoj, 3)
        emojis_img = io.imread('images/emojis/%s.png' % str(class_names[y_pred]))
        plt.imshow(emojis_img)
        plt.xlabel('Emoji Expression', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

# ----------
# Evaluation
# ----------

net = VGG('VGG19')
checkpoint = torch.load(weight_path)
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

with torch.no_grad():
    for names, x in test_loader:
        batch, ncrops, c, h, w = x.shape

        # crops into batch
        x = x.view(batch * ncrops, c, h, w)
        x = x.cuda()
        outputs = net(x)
        outputs = outputs.view(batch, ncrops, -1)
        outputs_avg = outputs.mean(1)
        score = F.softmax(outputs_avg, 1)
        predicted = torch.max(outputs_avg.data, 1)[1]  # torch.max returns a tuple (max_values, indices).

        for i in range(batch):
            y_pred = int(predicted[i].cpu().numpy())
            y_score = score[i].cpu().numpy()
            visualization(y_score, y_pred, names[i])
