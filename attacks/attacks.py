import logging
import time
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as Data
from sklearn.metrics import accuracy_score

from seed import seed_everything
from attacks.Attack import AttackError

from attacks.adaptive.Square import Square
from attacks.adaptive.NESScore import NESScore
from attacks.adaptive.HSJA import HSJA
from attacks.adaptive.QEBA import QEBA
from attacks.adaptive.SurFree import SurFree
from attacks.adaptive.Boundary import Boundary
from attacks.adaptive.DBAGP import DBAGP

import box


@torch.no_grad()
def natural_performance(model, loader):
    logging.info("Computing natural accuracy")
    y_true, y_pred = [], []
    pbar = tqdm(range(0, len(loader)), colour="red")
    for i, (x, y, p) in (enumerate(loader)):
        x, y = x.cuda(), y.cuda()
        start = time.time()
        logits, is_cache = model(x)
        end = time.time()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

        logging.info(f"True Label : {y[0]} | Predicted Label : {preds[0]} | is_cache : {is_cache[0]} | latency : {end - start}")

        if model.config["action"] == "rejection":
            preds = [preds[j] if not is_cache[j] else -1 for j in range(len(preds))]
        true = y.detach().cpu().numpy().tolist()
        y_true.extend(true)
        y_pred.extend(preds)
        pbar.update(1)
        pbar.set_description("Running accuracy: {} | hits : {}".format(accuracy_score(y_true, y_pred), model.cache_hits))
    logging.info("FINISHED")
    return accuracy_score(y_true, y_pred)


# @torch.no_grad()
def attack_loader(model, src_images, src_labels, tgt_images, tgt_labels, model_config, attack_config, args_dba):

    # Load attack
    try:
        attacker = globals()[attack_config['attack']](model, model_config, attack_config)
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')
    '''mse = attacker.compute_mse(src_images[0], tgt_images[0])
    print(mse)
    while True:
        x=1'''
    '''if attack_config['targeted']:
        target_labels = []
        for _, (_, y, p) in enumerate(loader):
            target_label = y.item()
            while target_label == y.item():
                target_label = np.random.randint(0, len(loader.dataset.targeted_dict))
            target_labels.append(target_label)
    else:
        target_labels = None'''

    # Run attack and compute adversarial accuracy
    y_true, y_pred = [], []
    #pbar = tqdm(loader, colour="yellow")
    #for i, (x, y, p) in enumerate(pbar):
    mses = []
    success = 0.
    queries = 0.
    total_cache_hits = 0.
    total_first = 0.
    total_queries = 0.
    for i in tqdm(range(tgt_images.shape[0])):
        #for i in tqdm(range(2)):
        '''x = x.cuda()
        y = y.cuda()'''
        seed_everything()
        try:
            if attack_config['targeted']:
                # x_adv = attacker.attack_targeted(x, y_target, x_adv_init)
                #yy = model.model(src_images[i].unsqueeze(0)).argmax(dim=1)
                #print(yy)  255
                #print(src_labels[i].unsqueeze(0)) 255
                #while True:
                #x=1
                '''print("tgt label and src label: ", end='')
                print(tgt_labels[i], end='   ')
                print(src_labels[i])'''
                x_adv = attacker.attack_targeted(tgt_images[i].unsqueeze(0), src_labels[i].unsqueeze(0),
                                                 src_images[i].unsqueeze(0), args_dba)
            else:
                # pass
                x_adv = attacker.attack_untargeted(src_images[i].unsqueeze(0), src_images[i].unsqueeze(0))
        except AttackError as e:
            print(e)
            x_adv = src_images[i].unsqueeze(0)

        x_adv = x_adv.cuda()
        logits = model.f(x_adv)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        true = tgt_labels[i].unsqueeze(0).detach().cpu().numpy().tolist()

        y_true.extend(true)
        y_pred.extend(preds)
        print("Running Accuracy: {} ".format(accuracy_score(y_true, y_pred)))
        mse = attacker.compute_mse(tgt_images[i], x_adv)
        mses.extend([mse])
        if mse < 0.005:
            success += 1.0
            queries += attacker.get_total_queries()
        total_cache_hits += attacker.get_cache_hits()
        total_first += attacker.get_first()
        total_queries += attacker.get_total_queries()
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Cache Hits / Total Queries : {attacker.get_cache_hits()} / {attacker.get_total_queries()}"
        )
        attacker.reset()
    attack_success_rate = success / tgt_images.shape[0]
    mean_mse = np.mean(mses)
    if success > 0:
        queries /= success
    print("MSEs: ", end='')
    print(mses)
    print("Attack Success Rate: ", end='')
    print(attack_success_rate)
    print("Mean MSE: ", end='')
    print(mean_mse)
    print("Mean Success Queries: ", end='')
    print(queries)
    print("Total Cache Hits: ", end='')
    print(total_cache_hits)
    print("Total Queries: ", end='')
    print(total_queries)
    print("TPR: ", end='')
    print(total_cache_hits / total_queries)
    print("Average Queries to Detect: ", end='')
    print(total_first / tgt_images.shape[0])
    logging.info("FINISHED")
