import random
from rich.progress import track
from torchvision import os
import yaml

import numpy as np
import torch
from constants import ATTR_TO_NUM, NUM_TO_ATTR
from torchvision import utils

from nets import LatentClassifierNet
from utils.functions import clip_image

random.seed(42)
data_directory = "./data/test/"
output_directory = "./prepared_data/test/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

classifier_model_path = "./models/latent_classifier_epoch_20.pth"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device("cuda")

classifier = LatentClassifierNet([9216, 2048, 512, 40], activation="leakyrelu")
classifier.load_state_dict(torch.load(classifier_model_path, map_location=DEVICE))
classifier.eval()
classifier.to(DEVICE)

config_file = '20_attrs'
config = yaml.safe_load(open('./configs/' + config_file + '.yaml', 'r'))
all_attributes = [ATTR_TO_NUM[a] for a in config["attr"].split(',')]

num_samples = 1000

def create_dataset(num_samples=num_samples, all_attributes=all_attributes):
    all_coeffs = np.zeros((num_samples, len(all_attributes)), dtype=np.int8)

    for i in range(num_samples):
        local_attributes = random.sample(range(len(all_attributes)), random.randint(1, len(all_attributes)))
        global_attributes = [all_attributes[a] for a in local_attributes]
        latent_code = np.load(data_directory + "latent_code_%05d.npy" % i)
        latent_code = torch.tensor(latent_code).to(DEVICE)

        predict_label = classifier(latent_code.view(latent_code.size(0), -1))
        label = torch.sigmoid(predict_label)
        attr_prob = label[torch.arange(label.shape[0]), global_attributes]
        coeff = torch.where(attr_prob > 0.5, -1, 1).detach().cpu().numpy()
        all_coeffs[i][local_attributes] = coeff

    np.save(output_directory + ".npy", all_coeffs)
    
    
if __name__ == "__main__":
    create_dataset(name=conf_file)
