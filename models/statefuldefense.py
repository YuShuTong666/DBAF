import random

import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision
from models.resnet import resnet152, resnet20
from models.iot_sqa import IOTSQAClassifier, IOTSQAEncoder
from abc import abstractmethod
from multiprocessing import Pool
import hashlib
from collections import Counter
from skimage.feature import local_binary_pattern
import cv2


class StateModule:
    @abstractmethod
    def getDigest(self, img):
        """
        Returns a digest of the image
        :param img:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resultsTopk(self, img, k):
        """
        Return a list of top k tuples (distance, prediction) - smallest distance first
        :param img:
        :param k:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resetCache(self):
        """
        Reset the cache
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, img, prediction):
        """
        Add an image to the cache
        :param img:
        :return:
        """
        raise NotImplementedError


# 10 10 10 24 25 10 10
# mse(x+c, qx)
class BlackFooler(StateModule):
    def __init__(self, arguments, args):
        self.window_size = arguments["window_size"]
        self.num_hashes_keep = arguments["num_hashes_keep"]
        self.round = arguments["round"]
        self.step_size = arguments["step_size"]
        self.step_size_accurate = arguments["step_size_accurate"]
        self.input_shape = arguments["input_shape"]
        self.salt_type = arguments["salt"]
        if self.salt_type and not args.nosalt:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)
        self.adap_c = args.adap_c
        #self.cache = {}
        self.cache_bad = {}
        self.inverse_cache = {}
        self.inverse_cache_accurate = {}
        self.input_idx = 0
        self.pool = Pool(processes=arguments["num_processes"])
        self.pool_accurate = Pool(processes=arguments["num_processes"])

    @staticmethod
    def hash_helper(arguments):
        img = arguments['img']
        idx = arguments['idx']
        window_size = arguments['window_size']
        return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    def preprocess(self, array, salt, round=1, normalized=True):
        if len(array.shape) != 3:
            raise Exception("expected 3d image")
        if normalized:
            # input image normalized to [0,1]
            array = np.array(array.cpu()) * 255.
        array = (array + salt) % 255.
        array = array.reshape(-1)
        min_dist = np.inf
        best_c = None
        if self.adap_c:  # 自适应策略
            array_cuda = torch.from_numpy(array).cuda()
            array_cuda = array_cuda.repeat(31, 1)
            c = torch.tensor(range(-15, 16)).reshape(31, 1).repeat(1, array_cuda.shape[1]).cuda()
            array_c = torch.clip(array_cuda + c, 0, 255)
            array_discretized = torch.round(array_c / round) * round
            dist = torch.sum(torch.abs(array_c - array_discretized), dim=1)
            best_c = dist.argmax().cpu().numpy() - 15
            '''array_cuda = torch.from_numpy(array).cuda()
            for c in range(-15, 15):
                array_c = torch.clip(array_cuda + c, 0, 255)
                array_discretized = torch.round(array_c / round) * round
                dist = torch.sum(torch.abs(array_c - array_discretized))
                if dist < min_dist:
                    min_dist = dist
                    best_c = c'''
            array = np.clip(array + best_c, 0, 255)
        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def preprocess_accurate(self, array, normalized=True):
        if len(array.shape) != 3:
            raise Exception("expected 3d image")
        ### np.round(array * 255 + 0.5)
        if normalized:
            # input image normalized to [0,1]
            array = np.array(array.cpu()) * 255.
        #array = (array / 2 + salt / 2) % 255.
        array = array.reshape(-1)
        #array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def getDigest(self, img):
        img = self.preprocess(img, self.salt, self.round)
        total_len = int(len(img))
        idx_ls = []

        for el in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(BlackLight.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def getDigestAccurate(self, img):
        img = self.preprocess_accurate(img)
        total_len = int(len(img))
        idx_ls = []

        for el in range(int((total_len - self.window_size + 1) / self.step_size_accurate)):
            idx_ls.append({"idx": el * self.step_size_accurate, "img": img, "window_size": self.window_size})
        hash_list = self.pool_accurate.map(BlackLight.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def resetCache(self):
        #self.cache = {}
        self.cache_bad = {}
        self.inverse_cache = {}
        self.inverse_cache_accurate = {}
        self.input_idx = 0
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

    def add(self, img, prediction):
        self.input_idx += 1
        hashes = self.getDigest(img)
        for el in hashes:
            if el not in self.inverse_cache:
                self.inverse_cache[el] = [self.input_idx]
            else:
                self.inverse_cache[el].append(self.input_idx)
        #self.cache[self.input_idx] = prediction

    def add_accurate(self, img, is_bad):
        self.input_idx += 1
        hashes = self.getDigestAccurate(img)
        for el in hashes:
            if el not in self.inverse_cache_accurate:
                self.inverse_cache_accurate[el] = [self.input_idx]
            else:
                self.inverse_cache_accurate[el].append(self.input_idx)
        self.cache_bad[self.input_idx] = is_bad

    def resultsTopk(self, img, k):
        #print("#####################################################################")
        hashes = self.getDigest(img)
        #print("Hashes: ", end='')
        #print(hashes)
        sets = list(map(self.inverse_cache.get, hashes))
        #print(sets)
        sets = [i for i in sets if i is not None]
        #print(sets)
        sets = [item for sublist in sets for item in sublist]
        #print(sets)
        if not sets:
            return []
        sets = Counter(sets)
        #print(sets)
        #print(sets.most_common(k))
        result = [((self.num_hashes_keep - x[1]) / self.num_hashes_keep, None) for x in sets.most_common(k)]
        #print(result)
        #print("#####################################################################")
        return result

    def resultsTopkAccurate(self, img, k):
        #print("#####################################################################")
        hashes = self.getDigestAccurate(img)
        #print("Hashes: ", end='')
        #print(hashes)
        sets = list(map(self.inverse_cache_accurate.get, hashes))
        #print(sets)
        sets = [i for i in sets if i is not None]
        #print(sets)
        sets = [item for sublist in sets for item in sublist]
        #print(sets)
        if not sets:
            return []
        sets = Counter(sets)
        #print(sets)
        #print(sets.most_common(k))
        result = [((self.num_hashes_keep - x[1]) / self.num_hashes_keep, self.cache_bad[x[0]]) for x in sets.most_common(k)]
        #print(result)
        #print("#####################################################################")
        return result


class BlackLight(StateModule):
    def __init__(self, arguments):
        self.window_size = arguments["window_size"]
        self.num_hashes_keep = arguments["num_hashes_keep"]
        self.round = arguments["round"]
        self.step_size = arguments["step_size"]
        self.input_shape = arguments["input_shape"]
        self.salt_type = arguments["salt"]
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

        #self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        self.pool = Pool(processes=arguments["num_processes"])

    @staticmethod
    def hash_helper(arguments):
        img = arguments['img']
        idx = arguments['idx']
        window_size = arguments['window_size']
        return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    def preprocess(self, array, salt, round=1, normalized=True):
        if len(array.shape) != 3:
            raise Exception("expected 3d image")
        if (normalized):
            # input image normalized to [0,1]
            array = np.array(array.cpu()) * 255.
        array = (array + salt) % 255.
        array = array.reshape(-1)

        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def getDigest(self, img):
        img = self.preprocess(img, self.salt, self.round)
        total_len = int(len(img))
        idx_ls = []

        for el in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(BlackLight.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def resetCache(self):
        #self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

    def add(self, img, prediction):
        self.input_idx += 1
        hashes = self.getDigest(img)
        for el in hashes:
            if el not in self.inverse_cache:
                self.inverse_cache[el] = [self.input_idx]
            else:
                self.inverse_cache[el].append(self.input_idx)
        #self.cache[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        sets = list(map(self.inverse_cache.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return []
        sets = Counter(sets)
        result = [((self.num_hashes_keep - x[1]) / self.num_hashes_keep, None) for x in sets.most_common(k)]
        return result


class OSDEncoder(torch.nn.Module):
    def __init__(self):
        super(OSDEncoder, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.drop1 = torch.nn.Dropout2d(0.25)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = torch.nn.Conv2d(64, 64, 3)
        self.drop2 = torch.nn.Dropout2d(0.25)

        self.fc1 = torch.nn.Linear(64 * 6 * 6, 512)
        self.drop3 = torch.nn.Dropout2d(0.5)
        self.fc2 = torch.nn.Linear(512, 256)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv4(x)), 2)
        x = self.drop2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 64 * 6 * 6)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


class OriginalStatefulDetector(StateModule):
    def __init__(self, arguments):
        self.encoder = OSDEncoder().cuda()
        checkpoint = torch.load(arguments["encoder_path"])
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.input_shape = arguments["input_shape"]
        if arguments["salt"] is not None:
            self.salt = arguments["salt"]
        else:
            self.salt = np.zeros(self.input_shape).astype(np.int16)

        self.cache = {}

    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        return self.encoder(img.to(next(self.encoder.parameters()).device).detach().unsqueeze(0)).squeeze(0)

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction

    def resultsTopk(self, img, k):
        # print(torch.min(img), torch.max(img))
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.linalg.norm(embed - query_embed).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)[:k]
        # top_dists = np.argpartition(dists, k - 1)
        result = [(dists[i], preds[i]) for i in top_dists]
        return result


class IOTSQA(StateModule):
    def __init__(self, arguments):
        self.encoder = IOTSQAEncoder()
        # self.encoder.load_weights("models/pretrained/iot_sqa_encoder.h5")
        self.encoder.eval()
        self.encoder = self.encoder.cuda()

        self.cache = {}

    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        return self.encoder(img.detach().unsqueeze(0)).squeeze(0)

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction

    def resultsTopk(self, img, k):
        # print(torch.min(img), torch.max(img))
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.linalg.norm(embed - query_embed).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)[:k]
        # top_dists = np.argpartition(dists, k - 1)
        result = [(dists[i], preds[i]) for i in top_dists]
        return result


class PIHA(StateModule):
    def __init__(self, arguments):
        super(PIHA, self).__init__()
        self.input_shape = arguments["input_shape"]
        self.block_size = arguments["block_size"]
        self.cache_predictions = {}
        self.input_idx = 0
        self.cache = self.getDigest(torch.zeros(arguments["input_shape"]))

    def _piha_hash(self, x):
        N = x.shape[2]
        # Image preprocessing
        x = x.cpu().numpy().transpose(1, 2, 0)
        x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

        # Color space transformation
        x_hsv = cv2.cvtColor(x_filtered, cv2.COLOR_RGB2HSV)

        # Use only H channel for HSV color space
        x_h = x_hsv[:, :, 0].reshape((N, N, 1))
        x_h = np.pad(x_h, ((0, self.block_size - N % self.block_size), (0, self.block_size - N % self.block_size), (0, 0)),
                     'constant')
        N = x_h.shape[0]

        # Block division and feature matrix calculation
        blocks_h = [
            x_h[i:i + self.block_size, j:j + self.block_size] for i in range(0, N, self.block_size)
            for j in range(0, N, self.block_size)
        ]
        features_h = np.array([np.sum(block) for block in blocks_h]).reshape((N // self.block_size, N // self.block_size))

        # Local binary pattern feature extraction
        features_lbp = local_binary_pattern(features_h, 8, 1)

        # Hash generation
        # hash_array = ''.join([f'{(int(_)):x}' for _ in features_lbp.flatten().tolist()])
        # hash_array = ''.join([format(int(_), '02x') for _ in features_lbp.flatten().tolist()])
        hash_array = features_lbp.flatten()
        hash_array = np.expand_dims(hash_array, axis=0)
        return hash_array

    def getDigest(self, img):
        h = self._piha_hash(img)
        return h

    def resetCache(self):
        self.cache = self.getDigest(torch.zeros(tuple(self.input_shape)))

    def add(self, img, prediction):
        self.input_idx += 1
        hash = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hash))
        self.cache_predictions[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hash = self.getDigest(img)
        hamming_dists = np.count_nonzero(hash != self.cache, axis=1) / self.cache.shape[1]
        closest = np.argsort(hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[closest != 0]
        if len(closest) == 0:
            return []
        result = [(hamming_dists[i], self.cache_predictions[i]) for i in closest]
        return result


class NoOpState(StateModule):
    def __init__(self, arguments):
        pass

    def getDigest(self, img):
        return img

    def resetCache(self):
        pass

    def add(self, img, prediction):
        pass

    def resultsTopk(self, img, k):
        return []


class StatefulClassifier(torch.nn.Module):
    def __init__(self, model, state_module, hyperparameters, args):
        super().__init__()
        self.config = hyperparameters
        self.model = model
        self.state_module = state_module
        ### 鲁棒碰撞检测阈值
        self.threshold = hyperparameters["threshold"]
        self.aggregation = hyperparameters["aggregation"]
        ### 检测到攻击后cache
        self.add_cache_hit = hyperparameters["add_cache_hit"]
        ### 检测到攻击后重置cache
        self.reset_cache_on_hit = hyperparameters["reset_cache_on_hit"]
        self.cache_hits = 0
        self.total = 0
        self.distances = []
        self.distances_accurate = []
        self.normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.transform_norm = torchvision.transforms.Compose([self.normalize])
        self.fooler = False
        self.eps = args.eps
        self.lr = args.lr
        self.no_hash = args.no_hash
        self.accurate = args.accurate
        self.first = 0
        self.celeba = args.celeba
        if hyperparameters["state"]["type"] == "blackfooler":
            self.fooler = True
            self.accurate = self.fooler
        # if not self.fooler:
        #     self.accurate = False

    def reset(self):
        self.state_module.resetCache()
        self.cache_hits = 0
        self.total = 0
        self.first = 0
        self.distances = []
        self.distances_accurate = []

    def forward_single(self, x):
        if self.celeba:
            x_norm = x
        else:
            x_norm = self.transform_norm(x)
        self.total += 1
        if self.no_hash:
            prediction = self.model(x_norm.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
            return prediction, False
        cached_prediction = None
        similar = False
        similar_accurate = False
        if self.aggregation == 'closest':
            if self.accurate:
                similarity_result_accurate = self.state_module.resultsTopkAccurate(x, 1)
                is_bad = False
                if len(similarity_result_accurate) > 0:
                    dist, is_bad = similarity_result_accurate[0]
                    self.distances_accurate.append(dist)
                    if dist == 0:
                        similar_accurate = True
                    #print(is_bad)
                if similar_accurate and is_bad is False:
                    #print("similar_accurate")
                    prediction = self.model(x_norm.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
                    return prediction, False

            similarity_result = self.state_module.resultsTopk(x, 1)
            if len(similarity_result) > 0:
                dist, cached_prediction = similarity_result[0]
                self.distances.append(dist)
                if dist <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        elif self.aggregation == 'average':
            similarity_result = self.state_module.resultsTopk(x, self.config['num_to_average'])
            if len(similarity_result) >= self.config['num_to_average']:
                dist, cached_prediction = similarity_result[0]
                dists = [dist for (dist, _) in similarity_result]
                if np.mean(dists) <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        prediction = self.model(x_norm.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        #if similar_accurate:
        #return prediction, True
        if self.lr == 666:
            pred_copy = torch.softmax(prediction.clone().detach_(), dim=1)
            # print(pred_copy)
            ya = torch.argmax(pred_copy, dim=1)
            a_value = pred_copy[0][ya]
            pred_copy[0][ya] = 0
            yb = torch.argmax(pred_copy, dim=1)
            b_value = pred_copy[0][yb]
            '''print("a_value: ", end='')
            print(a_value, end=' ||| ')
            print("b_value: ", end='')
            print(b_value)'''
        if similar:
            #if self.config["action"] != 'rejection_silent':
            # cached_prediction = -1 * torch.ones_like(cached_prediction)
            #print("similar")
            if self.fooler:
                '''print(prediction.shape)
                print(prediction)
                print(torch.topk(prediction.clone().detach_(), 10).values)'''
                pred_copy = torch.softmax(prediction.clone().detach_(), dim=1)
                ya = torch.argmax(pred_copy, dim=1)
                a_value = pred_copy[0][ya]
                pred_copy[0][ya] = 0
                yb = torch.argmax(pred_copy, dim=1)
                b_value = pred_copy[0][yb]
                ### 判定样本是否在决策边界
                if torch.abs(a_value - b_value) < self.eps:
                    y_dim = prediction.shape[1]
                    #prediction[0][torch.randint(0, y_dim, (1,))] = 100
                    # print(torch.abs(a_value - b_value))
                    #prediction[0][torch.argmax(prediction, dim=1)] = 0
                    #prediction[0][torch.argmax(prediction, dim=1)] = 0
                    #print(prediction.argmax(dim=1))
                    if self.first == 0:
                        self.first = self.total
                    if self.accurate and not similar_accurate:
                        self.state_module.add_accurate(x, True)
                    return prediction, True
                else:
                    self.cache_hits -= 1
                    if self.accurate and not similar_accurate:
                        self.state_module.add_accurate(x, False)
                    return prediction, False
            if self.first == 0:
                self.first = self.total
            if self.accurate and not similar_accurate:
                self.state_module.add_accurate(x, True)
            return prediction, True

            #return cached_prediction.cuda(), True

            #prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
            #return prediction, False
        #prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        self.state_module.add(x, prediction.detach().cpu())
        if self.accurate and not similar_accurate:
            self.state_module.add_accurate(x, False)
        return prediction, False

    def forward_batch(self, x):
        batch_size = x.shape[0]
        logits, is_cache = [], []
        for i in range(batch_size):
            pred, is_cached = self.forward_single(x[i])
            logits.append(pred)
            is_cache.append(is_cached)
        logits = torch.cat(logits, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs, is_cache

    def forward(self, x):
        if len(x.shape) == 3:
            return self.forward_single(x)
        else:
            return self.forward_batch(x)

    def f(self, x):
        if len(x.shape) == 3:
            prob, is_cache = self.forward_single(x)
        else:
            prob, is_cache = self.forward_batch(x)
        return prob


def init_stateful_classifier_v1(config):
    if config['architecture'] == 'resnet20':
        model = torch.load("models/pretrained/resnet20-12fca82f-single.pth", map_location="cpu")
        model.eval()
    elif config['architecture'] == 'resnet152':
        model = resnet152()
        model.eval()
    elif config['architecture'] == 'iot_sqa':
        model = IOTSQAClassifier()
        # model.load_weights("models/pretrained/iot_sqa_classifier.h5")
        model.eval()
    elif config['architecture'] == 'celebahq':

        class CelebAHQClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 307)
                self.model.load_state_dict(
                    torch.load(
                        "models/pretrained/facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"))
                self.xform = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            def forward(self, x):
                x = self.xform(x)
                return self.model(x)

        model = CelebAHQClassifier()
        model.eval()
    else:
        raise NotImplementedError("Architecture not supported.")

    if config["state"]["type"] == "blacklight":
        state_module = BlackLight(config["state"])
    elif config["state"]["type"] == "PIHA":
        state_module = PIHA(config["state"])
    elif config["state"]["type"] == "OSD":
        state_module = OriginalStatefulDetector(config["state"])
    elif config["state"]["type"] == "iot_sqa":
        state_module = IOTSQA(config["state"])
    elif config["state"]["type"] == "no_op":
        state_module = NoOpState(config["state"])
    else:
        raise NotImplementedError("State module not supported.")

    return StatefulClassifier(model, state_module, config)


def init_stateful_classifier_v2(config, model, args):
    if config["state"]["type"] == "blacklight":
        state_module = BlackLight(config["state"])
    elif config["state"]["type"] == "PIHA":
        state_module = PIHA(config["state"])
    elif config["state"]["type"] == "OSD":
        state_module = OriginalStatefulDetector(config["state"])
    elif config["state"]["type"] == "iot_sqa":
        state_module = IOTSQA(config["state"])
    elif config["state"]["type"] == "no_op":
        state_module = NoOpState(config["state"])
    elif config["state"]["type"] == "blackfooler":
        state_module = BlackFooler(config["state"], args)
    else:
        raise NotImplementedError("State module not supported.")

    return StatefulClassifier(model, state_module, config, args)
