import os
import sys
import warnings
import logging
import logging.handlers

# import multiprocessing
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

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="lr.")
parser.add_argument("--seed", type=int, default=123, help="Random seed.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=2, help="Number of hidden units.")
parser.add_argument("--n_img", type=int, default=50, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.1, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--epochs", type=int, default=500, help="Number of epochs to train."
)
parser.add_argument("--k", type=int, default=5000, help="Number of nodes to train.")
parser.add_argument("--celeba", action="store_true", help="celeba")
parser.add_argument("--maxq", type=int, default=5000, help="Number of max queries.")
parser.add_argument(
    "--method",
    default="HSJA",
    help='Options: "GM", "HSJA", "QEBA", "GF", "QEBAM", "JBF", "GIF", "RBD".',
)
parser.add_argument(
    "--layers", type=int, default=16, help="Number of graph filter layers."
)
parser.add_argument(
    "--alpha", type=float, default=0.0, help="Initial residual connection."
)
parser.add_argument(
    "--T", type=float, default=0.0002, help="Initial residual connection."
)
parser.add_argument(
    "--delta", type=float, default=0.8, help="Initial residual connection."
)
parser.add_argument(
    "--model", type=str, default="resnet", help="Option: resnet, vgg, inception"
)
parser.add_argument("--r", type=int, default=3, help=".")
parser.add_argument("--iteration", type=int, default=8, help=".")
parser.add_argument("--init", type=int, default=100, help=".")
parser.add_argument("--bf", action="store_true", default=False, help="bi filter.")
parser.add_argument("--time", action="store_true", default=False, help="Time prior.")
parser.add_argument(
    "--ntime", action="store_true", default=False, help="Naive Time Prior."
)
parser.add_argument("--disable_logging", action="store_true")
parser.add_argument("--num_images", type=int, default=50)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--bright", action="store_true")
parser.add_argument("--adap_c", action="store_true")
parser.add_argument("--eps", type=float, default=0.1, help="eps of ")
parser.add_argument("--nosalt", action="store_true")
parser.add_argument("--random_init", action="store_true")
parser.add_argument("--no_hash", action="store_true")
parser.add_argument("--accurate", action="store_true")
parser.add_argument("--theta_min", type=float, default=1.7e-5, help="theta for surfree")
# parser.add_argument('--config', type=str, default='configs/imagenet/blacklight/hsja/targeted/fooler/config.json')
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--data", type=str, default="imagenet")
parser.add_argument("--defense", type=str, default="DPF")
parser.add_argument("--attack", type=str, default="dbagp")
parser.add_argument("--untargeted", action="store_true")
parser.add_argument("--normal", action="store_true")
args_dba = parser.parse_args()



def imagenet_target_attack(args, N_img, model_config):
    import torchvision.models as models

    if args.model == "resnet":
        model = models.resnet50(pretrained=True).eval()
        print("Attacked model: resnet50")
    elif args.model == "vgg":
        model = models.vgg16(pretrained=True).eval()
        print("Attacked model: vgg16")
    elif args.model == "inception":
        model = models.inception_v3(pretrained=True).eval()
        print("Attacked model: inception_v3")
    elif args.model == "res18":
        model = models.resnet18(pretrained=True).eval()
        print("Attacked model: resnet18")
    elif args.model == "densenet":
        model = models.densenet121(pretrained=True).eval()
        print("Attacked model: densenet")
    elif args.model == "googlenet":
        model = models.googlenet(pretrained=True).eval()
        print("Attacked model: googlenet")
    elif args.model == "vit":
        model = models.vit_b_16(pretrained=True).eval()
    elif args.model == "bitdepth":
        model = BitDepthModel(pretrained=True).eval()
    else:
        raise Exception("Invalid model name!")
    model.cuda()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    root = 'path/to/imagenet val'

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
    )
    src_images, src_labels = None, None
    tgt_images, tgt_labels = None, None
    # while (len(src_images) < N_img):
    for _, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        if (
            y[0] != y[1]
            and torch.argmax(model.f(x[0].unsqueeze(0)).squeeze()) == y[0]
            and torch.argmax(model.f(x[1].unsqueeze(0)).squeeze()) == y[1]
        ):
            if src_images is None:
                src_images, src_labels = x[0].unsqueeze(0), y[0].unsqueeze(0)
                tgt_images, tgt_labels = x[1].unsqueeze(0), y[1].unsqueeze(0)
            else:
                src_images = torch.cat((src_images, x[0].unsqueeze(0)))
                src_labels = torch.cat((src_labels, y[0].unsqueeze(0)))
                tgt_images = torch.cat((tgt_images, x[1].unsqueeze(0)))
                tgt_labels = torch.cat((tgt_labels, y[1].unsqueeze(0)))

        if src_images is None:
            continue
        if src_images.shape[0] == N_img:
            break
    print("data loaded!")
    return src_images, src_labels, tgt_images, tgt_labels, model


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CelebAModel(nn.Module):
    def __init__(self, num_class, model="resnet", pretrained=False, gpu=True):
        super(CelebAModel, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu
        self.num_class = num_class
        self.model_name = model
        # self.resnet = models.resnet18(pretrained=pretrained)
        # self.resnet = models.wide_resnet50_2(pretrained=pretrained)
        if model == "resnet":
            self.model = models.resnet50(
                pretrained=pretrained, num_classes=self.num_class
            )
            print("Train model: resnet50")
        elif model == "vgg":
            self.model = models.vgg16_bn(
                pretrained=pretrained, num_classes=self.num_class
            )
            print("Train model: vgg16")
        elif model == "inception":
            self.model = models.inception_v3(
                pretrained=pretrained,
                num_classes=self.num_class,
                transform_input=True,
                aux_logits=False,
            )
            print("Train model: inception_v3")
        elif model == "vit":
            self.model = models.vit_b_16()
            self.output = nn.Linear(1000, self.num_class)
        else:
            raise Exception("Invalid model name!")

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.model(x)
        if self.model_name == "vit":
            x = self.output(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


def celeba_target_attack(args, N_img, model_config):
    model_root = 'path/to/your model'
    model = CelebAModel(100, model=args.model, pretrained=False, gpu=True)
    if args.model == "resnet":
        stat = torch.load(model_root + "resnetceleba.model")
        model.load_state_dict(stat)
        print("Attacked model: resnet50")
    elif args.model == "vit":
        stat = torch.load(model_root + "vitceleba.model")
        model.load_state_dict(stat)
    else:
        raise Exception("Invalid model name!")
    model.eval()
    model.cuda()
    root = 'path/to/celeba'
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    from celeba_dataset import CelebADataset

    testset = CelebADataset(
        root_dir=root,
        is_train=False,
        transform=transform,
        preprocess=False,
        random_sample=False,
        n_id=100,
    )
    loader = Data.DataLoader(
        dataset=testset,
        batch_size=2,
        shuffle=True,
    )
    src_images, src_labels = None, None
    tgt_images, tgt_labels = None, None
    for _, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        if (
            y[0] != y[1]
            and torch.argmax(model.f(x[0].unsqueeze(0)).squeeze()) == y[0]
            and torch.argmax(model.f(x[1].unsqueeze(0)).squeeze()) == y[1]
        ):
            if src_images is None:
                src_images, src_labels = x[0].unsqueeze(0), y[0].unsqueeze(0)
                tgt_images, tgt_labels = x[1].unsqueeze(0), y[1].unsqueeze(0)
            else:
                print(src_labels.shape)
                print(y[0].unsqueeze(0).shape)
                src_images = torch.cat((src_images, x[0].unsqueeze(0)))
                src_labels = torch.cat((src_labels, y[0].unsqueeze(0)))
                tgt_images = torch.cat((tgt_images, x[1].unsqueeze(0)))
                tgt_labels = torch.cat((tgt_labels, y[1].unsqueeze(0)))
        else:
            print("not fuhe!", end=" ")
            print(_)
        if src_images is None:
            continue
        if src_images.shape[0] == N_img:
            break
    return src_images, src_labels, tgt_images, tgt_labels, model


def main(args):
    # Set up logging and load config.
    if not args.disable_logging:
        random.seed(0)
        log_dir = os.path.join(
            "/".join(args.config.split("/")[:-1]),
            "logs",
            args.config.split("/")[-1].split(".")[0],
        )
        writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(
                writer.log_dir,
                f"log_{args.start_idx}_{args.start_idx + args.num_images}.txt",
            ),
            level=logging.INFO,
        )
        logging.info(args)

    config = json.load(open(args.config))
    model_config, attack_config = config["model_config"], config["attack_config"]

    logging.info(model_config)
    logging.info(attack_config)

    # Load model.
    if args.celeba:
        src_images, src_labels, tgt_images, tgt_labels, model = celeba_target_attack(
            args=args_dba, N_img=args_dba.n_img, model_config=model_config
        )
    else:
        src_images, src_labels, tgt_images, tgt_labels, model = imagenet_target_attack(
            args=args_dba, N_img=args_dba.n_img, model_config=model_config
        )
    model.eval()
    model.to("cuda")
    # Load dataset.
    """if model_config["dataset"] == "mnist":
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
        raise ValueError("Dataset not supported.")"""
    """test_dataset = datasets.StatefulDefenseDataset(name=model_config["dataset"], transform=transform,
                                                   size=args.num_images, start_idx=args.start_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)"""

    if attack_config["attack"] == "natural_accuracy":
        # natural_performance(model, test_loader)
        pass
    else:
        attack_loader(
            model,
            src_images,
            src_labels,
            tgt_images,
            tgt_labels,
            model_config,
            attack_config,
            args_dba,
        )


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    main(args_dba)
