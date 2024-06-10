import argparse
import os
import numpy as np
import torch
import yaml
from rich.progress import track
import sys
from PIL import Image


sys.path.append('../')
from datasets import *
from models.multi_trainer import Trainer as MultiTrainer
from models.single_trainer import Trainer as SingleTrainer
from models.interface_trainer import Trainer as InterFaceTrainer
from models.tedi_trainer import Trainer as TediTrainer
from models.styleclip_trainer import Trainer as StyleCLIPTrainer
from utils.functions import *
from constants import ATTR_TO_NUM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to the config file.")
parser.add_argument("--out", type=str, default="./data/output/", help="Name of the out folder")
parser.add_argument("--model_path", type=str, default="", help="Path to the foulder of the desired model")
opts = parser.parse_args()

model = "20_attrs"
testdata_dir = "./prepared_data/test/"
n_steps = 11
scale = 2.0
n_samples = 300

log_dir_single = os.path.join(opts.log_path, "original_train") + "/"
save_dir = os.path.join("./outputs/evaluation/", opts.out) + "/"
os.makedirs(save_dir, exist_ok=True)
log_dir = os.path.join(opts.log_path, opts.config) + "/"
config = yaml.safe_load(open("./configs/" + opts.config + ".yaml", "r"))
attrs = config["attr"].split(",")
attr_num = [ATTR_TO_NUM[a] for a in attrs]


def get_trainer(model=0, config=config, log_dir=log_dir, attr_num=attr_num, attrs=attrs):
    if model == 0: #multi
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.load_model_multi(log_dir, model)
    elif model == 1: #singlw
        trainer = SingleTrainer(config, None, None, opts.label_file)
    elif model == 2: #interfaceGAN
        trainer = InterFaceTrainer(config, attr_num, attrs, opts.label_file)
    elif model == 3: #tediGAN
        trainer = TediTrainer(config, attr_num, attrs, opts.label_file)
    else: #styleCLIP
        trainer = StyleCLIPTrainer(config, attr_num, attrs, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
    trainer.to(DEVICE)
    return trainer


def apply_transformation(trainer, w_0, coeff, attrs=attrs, model=0):
    w_1 = w_0
    
    if model == 0: #single
        w_prev = w_0
        for i, c in enumerate(coeff):
            if c == 0:
                continue
            trainer.attr_num = ATTR_TO_NUM[attrs[i]]
            trainer.load_model(log_dir_single)
            trainer.to(DEVICE)

            w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
            w_1 = w_1.view(w_0.size())
            w_prev = w_1

    elif model == 1: # multi
        w_1 = trainer.T_net(
            w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE), scaling=1
        )
        w_1 = w_1.view(w_0.size())
    
            
    elif model == 2: #interfaceGAN
        w_prev = w_0
        for i, c in enumerate(coeff):
            if c == 0:
                continue
            trainer.attr_num = ATTR_TO_NUM[attrs[i]]
            trainer.load_model(log_dir_single)
            trainer.to(DEVICE)

            w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
            w_1 = w_1.view(w_0.size())
            w_prev = w_1
    
    elif model == 3: #tediGAN
        w_prev = w_0
        for i, c in enumerate(coeff):
            if c == 0:
                continue
            trainer.attr_num = ATTR_TO_NUM[attrs[i]]
            trainer.load_model(log_dir_single)
            trainer.to(DEVICE)

            w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
            w_1 = w_1.view(w_0.size())
            w_prev = w_1

    else: #styleCLIP
        w_prev = w_0
        for i, c in enumerate(coeff):
            if c == 0:
                continue
            trainer.attr_num = ATTR_TO_NUM[attrs[i]]
            trainer.load_model(log_dir_single)
            trainer.to(DEVICE)

            w_1 = trainer.T_net(w_prev.view(w_0.size(0), -1), c.unsqueeze(0))
            w_1 = w_1.view(w_0.size())
            w_prev = w_1

    return w_1


def get_ratios_from_sample(w_0, w_1, coeff, trainer):
    predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
    lbl_0 = torch.sigmoid(predict_lbl_0)
    attr_pb_0 = lbl_0[:, attr_num]
    predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
    lbl_1 = torch.sigmoid(predict_lbl_1)
    attr_pb_1 = lbl_1[:, attr_num]
    ratio = get_target_change(attr_pb_0, attr_pb_1, coeff, mean=False)
    return ratio


def metrics(model=0, attr=None, attr_i=None, orders=None, n_samples=n_samples,
                                     return_mean=True, n_steps=n_steps, scale=scale):
    with torch.no_grad():
        trainer = get_trainer(model)

        if (attr and attr_i is not None) or orders is not None:
            all_coeffs = np.load(testdata_dir + "labels/all.npy")
        else:
            all_coeffs = np.load(testdata_dir + "labels/overall.npy")

        class_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        ident_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))
        attr_ratios = np.zeros((n_samples, torch.linspace(0, scale, n_steps).shape[0]))

        for k in track(range(n_samples), track_title):
            w_0 = np.load(testdata_dir + "latent_code_%05d.npy" % k)
            w_0 = torch.tensor(w_0).to(DEVICE)

            predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
            lbl_0 = torch.sigmoid(predict_lbl_0)
            attr_pb_0 = lbl_0[:, attr_num]

            if attr and attr_i is not None:
                coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
                coeff[attr_i] = all_coeffs[k][attr_i]
            elif orders is not None:
                coeff = torch.zeros(all_coeffs[k].shape).to(DEVICE)
                coeff[orders[k]] = torch.tensor(all_coeffs[k][orders[k]], dtype=torch.float).to(DEVICE)
            else:
                coeff = torch.tensor(all_coeffs[k]).to(DEVICE)

            scales = torch.linspace(0, scale, n_steps).to(DEVICE)
            range_coeffs = coeff * scales.reshape(-1, 1)
            for i, alpha in enumerate(range_coeffs):
                w_1 = apply_transformation(
                    trainer=trainer, w_0=w_0, coeff=alpha, model=model
                )

                predict_lbl_1 = trainer.Latent_Classifier(w_1.view(w_0.size(0), -1))
                lbl_1 = torch.sigmoid(predict_lbl_1)
                attr_pb_1 = lbl_1[:, attr_num]

                ident_ratio = trainer.MSEloss(w_1, w_0)
                attr_ratio = get_attr_change(lbl_0, lbl_1, coeff, attr_num)
                class_ratio = get_target_change(attr_pb_0, attr_pb_1, coeff)

                class_ratios[k][i] = class_ratio
                ident_ratios[k][i] = ident_ratio
                attr_ratios[k][i] = attr_ratio

        if return_mean:
            class_r = class_ratios.mean(axis=0)
            recons = ident_ratios.mean(axis=0)
            attr_r = attr_ratios.mean(axis=0)
        else:
            class_r = class_ratios
            recons = ident_ratios
            attr_r = attr_ratios

        return class_r, recons, attr_r
    
    
if __name__ == "__main__":
    metrics()