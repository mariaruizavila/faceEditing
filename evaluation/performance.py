import time
import argparse
import os
import numpy as np
import torch
import torch.utils.data as data
import yaml
from rich.progress import track
from PIL import Image
import sys

from constants import ATTR_TO_NUM, CLASSIFIER, LABEL_FILE, LATENT_PATH, LOG_DIR, STYLEGAN

sys.path.append('../')
from models.multi_trainer import Trainer as MultiTrainer
from models.single_trainer import Trainer as SingleTrainer
from models.interface_trainer import Trainer as InterFaceTrainer
from models.tedi_trainer import Trainer as TediTrainer
from models.styleclip_trainer import Trainer as StyleCLIPTrainer

torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "Turing"
    DEVICE = torch.device('cpu')



log_dir = os.path.join(LOG_DIR, "performance") + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

config = yaml.safe_load(open('./configs/' + "performance" + '.yaml', 'r'))
attrs = config['attr'].split(',')
batch_size = config['batch_size']
epochs = config['epochs']

dlatents = np.load(LATENT_PATH)
w = torch.tensor(dlatents).to(DEVICE)

dataset_A = LatentDataset(LATENT_PATH, LABEL_FILE, training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)

attr_num = [ATTR_TO_NUM[a] for a in attrs]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="new_train", help="Path to the config file.")
parser.add_argument("--out", type=str, default="test", help="Name of the out folder")
opts = parser.parse_args()

def get_trainer(model=0, config=config, log_dir=log_dir, attr_num=attr_num, attrs=attrs):
    if model == 0: #single
        trainer = SingleTrainer(config, None, None, opts.label_file)
    elif model == 1: #multi
        trainer = MultiTrainer(config, attr_num, attrs, opts.label_file)
        trainer.load_model_multi(log_dir, model)
    elif model == 2: #interfaceGAN
        trainer = InterFaceTrainer(config, attr_num, attrs, opts.label_file)
    elif model == 3: #tediGAN
        trainer = TediTrainer(config, attr_num, attrs, opts.label_file)
    else: #styleCLIP
        trainer = StyleCLIPTrainer(config, attr_num, attrs, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
    trainer.to(DEVICE)
    return trainer

def get_performance(max_iters=100000, verbose=False):

    max_memories = []
    iteration_times = []
    file_sizes = []

    for n_attrs in range(0,21,4):
        n = n_attrs if n_attrs != 0 else 1
        attr_num_n = attr_num[:n]
        attrs_n = attrs[:n]

        trainer = get_trainer(0)
        trainer.initialize(STYLEGAN, CLASSIFIER)   
        trainer.to(DEVICE)

        times = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        total_iter = 0
        glob_time = time.time()
        for n_iter, list_A in track(enumerate(loader_A), f"Training model with {n} attributes..."):
            t = time.time()
            w_A, lbl_A = list_A
            w_A, lbl_A = w_A.to(DEVICE), lbl_A.to(DEVICE)
            trainer.update(w_A, None, n_iter)
            total_iter += 1

            e_time = time.time() - t
            times.append(e_time)

            if n_iter == max_iters:
                break

        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2 
        trainer.save_model_multi(log_dir, name="performance")
        model_size = os.path.getsize(log_dir + "tnet_performance.pth.tar") / 1024**2
        os.remove(log_dir + "tnet_performance.pth.tar")

        iteration_times.append(np.mean(times))
        max_memories.append(max_memory_allocated)
        file_sizes.append(model_size)

        if verbose:
            print(f"----- RESULTS FOR {n} ATTRIBUTES -----")
            print(f"Time in 100 iterations: {time.time() - glob_time}")
            print(f"Iteration mean time for 100 iterations (per iteration): {np.mean(times)}")
            print(f"Maximum GPU memory allocated: {max_memory_allocated:.2f} MB")
            #print(f"Model output file size: {model_size:.2f} MB")
            print(f"--------------------------------------")

    return iteration_times, max_memories, file_sizes

if __name__ == "__main__":
    max_iters = 300
    single_times, single_memories, single_files = get_performance(max_iters=max_iters, verbose=True)
