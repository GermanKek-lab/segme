import os
from skimage.transform import resize
from torch.utils.data import DataLoader
from skimage.io import imread
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from SegNet import SegNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_segmentation', map_location=device)

directory = input()
files = []
files += os.listdir(directory)
files.sort()

images_prediction = []
lesions_prediction = []

for path in files:
    images_prediction.append(imread(os.path.join(directory, path)))
    lesions_prediction.append(None)


size = (256, 256)
X_prediction = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images_prediction]
# X_prediction.reverse()
Y_prediction = [y for y in lesions_prediction]

X_prediction = np.array(X_prediction, np.float32)
Y_prediction = np.array(Y_prediction, np.float32)

batch_size = len(X_prediction)
tr = [i for i in range(len(X_prediction))]
data_tr_prediction = DataLoader(list(np.rollaxis(X_prediction[tr], 3, 1)),
                                batch_size=batch_size, shuffle=False)

model.eval()
pred_list = []
for i in data_tr_prediction:
    i = i.to(device)
    y_pred_prediction = (model(i).to("cpu").detach()) > 0.5
    pred_list.append(y_pred_prediction)
    plt.figure(figsize=(6, 6))
    i = i.to('cpu')
    for k in range(len(X_prediction)):
        save_image(y_pred_prediction[k, 0].float(), f"output/Tets{k + 1}.png")