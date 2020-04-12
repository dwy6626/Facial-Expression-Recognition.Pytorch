"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage import io

import os

from models import *
from evaluate import ImageDataset

input_dir = "inputs"
output_dir = "outputs"
batch_size = 200

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

def visualization(score, y_pred, fname):
    raw_img = io.imread(os.path.join(input_dir, fname))

    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes = plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    
    for i in range(len(class_names)):
        plt.bar(ind[i], score[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes = plt.subplot(1, 3, 3)
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
