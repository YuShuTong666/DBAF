import os
import warnings
import logging
import logging.handlers
#import multiprocessing
import random
import json
import socket
from datetime import datetime
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from attacks.attacks import *
from models.statefuldefense import init_stateful_classifier_v2
from utils import datasets
import argparse
from torchvision import datasets
import torch.utils.data as Data
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=666, help='lr.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2, help='Number of hidden units.')
parser.add_argument('--n_img', type=int, default=50, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--k', type=int, default=5000, help='Number of nodes to train.')
parser.add_argument('--celeba', action='store_true', help='celeba')
parser.add_argument('--maxq', type=int, default=5000, help='Number of max queries.')
parser.add_argument('--method', default='HSJA', help='Options: "GM", "HSJA", "QEBA", "GF", "QEBAM", "JBF", "GIF", "RBD".')
parser.add_argument('--layers', type=int, default=16, help='Number of graph filter layers.')
parser.add_argument('--alpha', type=float, default=0.0, help='Initial residual connection.')
parser.add_argument('--T', type=float, default=0.0002, help='Initial residual connection.')
parser.add_argument('--delta', type=float, default=0.8, help='Initial residual connection.')
parser.add_argument('--model', type=str, default="resnet", help='Option: resnet, vgg, inception')
parser.add_argument('--r', type=int, default=3, help='.')
parser.add_argument('--iteration', type=int, default=8, help='.')
parser.add_argument('--init', type=int, default=100, help='.')
parser.add_argument('--bf', action='store_true', default=False, help='bi filter.')
parser.add_argument('--time', action='store_true', default=False, help='Time prior.')
parser.add_argument('--ntime', action='store_true', default=False, help='Naive Time Prior.')
parser.add_argument('--disable_logging', action='store_true')
parser.add_argument('--config', type=str, default='configs/imagenet/blacklight/hsja/targeted/fooler/config.json')
parser.add_argument('--num_images', type=int, default=50)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--bright', action='store_true')
parser.add_argument('--adap_c', action='store_true')
parser.add_argument('--eps', type=float, default=0.2, help='eps of ')
parser.add_argument('--nosalt', action='store_true')
args_dba = parser.parse_args()

def imagenet_target_attack(args, N_img, model_config):
    import torchvision.models as models
    #model = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    if args.model == 'resnet':
        model = models.resnet50(pretrained=True).eval()
        print("Attacked model: resnet50")
    else:
        raise Exception("Invalid model name!")
    #if args.use_gpu:
    model.cuda()
    model = init_stateful_classifier_v2(model_config, model, args)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    #fmodel = box.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    root = 'box/data'
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    src_images = torchvision.io.read_image(root+'/src19.jpg')/255
    src_labels = torch.argmax(model.f(src_images).squeeze())
    tgt_images = torchvision.io.read_image(root+'/tgt19.jpg')/255
    tgt_labels = torch.argmax(model.f(tgt_images).squeeze())
    #while (len(src_images) < N_img):
    return src_images.unsqueeze(0), src_labels.unsqueeze(0), tgt_images.unsqueeze(0), tgt_labels.unsqueeze(0), model



def main(args):
    # Set up logging and load config.
    if not args.disable_logging:
        random.seed(0)
        log_dir = os.path.join("/".join(args.config.split("/")[:-1]), 'logs', args.config.split("/")[-1].split(".")[0])
        writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(writer.log_dir, f'log_{args.start_idx}_{args.start_idx + args.num_images}.txt'),
            level=logging.INFO)
        logging.info(args)

    config = json.load(open(args.config))
    model_config, attack_config = config["model_config"], config["attack_config"]

    logging.info(model_config)
    logging.info(attack_config)

    # Load model.

    src_images, src_labels, tgt_images, tgt_labels, model = imagenet_target_attack(args=args_dba, N_img=args_dba.n_img, model_config=model_config)
    model.eval()
    model.to("cuda")
    # Load dataset.
    '''if model_config["dataset"] == "mnist":
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    elif model_config["dataset"] == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "gtsrb":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "imagenet":
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    elif model_config["dataset"] == "iot_sqa":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "celebahq":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Dataset not supported.")'''

    '''test_dataset = datasets.StatefulDefenseDataset(name=model_config["dataset"], transform=transform,
                                                   size=args.num_images, start_idx=args.start_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)'''

    if attack_config["attack"] == "natural_accuracy":
        # natural_performance(model, test_loader)
        pass
    else:
        attack_loader(model, src_images, src_labels, tgt_images, tgt_labels, model_config, attack_config, args_dba)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    main(args_dba)
