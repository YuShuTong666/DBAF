import logging
from typing import Union, Any, Optional, Callable, List

import torch
from typing_extensions import Literal

import math

import eagerpy as ep
import numpy as np

from box.attacks import LinearSearchBlendedUniformNoiseAttack
from box.tensorboard import TensorBoard
from ..models import Model

from ..criteria import Criterion

from ..distances import l1

from ..devutils import atleast_kd, flatten

from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from ..distances import l2, linf
from typing import Callable, Union, Optional, Tuple, List, Any, Dict
import cv2
from ..tools import skip, ResNet
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
imsize = -1
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
input_depth = 32
PLOT = True
INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net'
KERNEL_TYPE = 'lanczos2'
LR = 0.01
tv_weight = 0.0
OPTIMIZER = 'adam'
mse = torch.nn.MSELoss().type(dtype)
cs = torch.nn.CosineSimilarity(0).type(dtype)
exp_weight=0.99
NET_TYPE = 'skip' # UNet, ResNet
def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)

    return res1 + res2
class DEEP_HSJA(MinimizationAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities [#Chen19].

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        steps : Number of optimization steps within each binary search step.
        initial_gradient_eval_steps: Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_gradient_eval_steps : Maximum number of evaluations for gradient estimation.
        stepsize_search : How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma : The binary search threshold theta is gamma / d^1.5 for
                   l2 attack and gamma / d^2 for linf attack.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        constraint : Norm to minimize, either "l2" or "linf"

    References:
        .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
        "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
        https://arxiv.org/abs/1904.02144
    """

    distance = l1

    def __init__(
        self,
        args,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: Union[
            Literal["geometric_progression"], Literal["grid_search"]
        ] = "geometric_progression",
        gamma: float = 1.0,
        tensorboard: Union[Literal[False], None, str] = False,
        constraint: Union[Literal["linf"], Literal["l2"]] = "l2",
        max_queries: int = 1000,
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.args = args
        self.init_attack = init_attack
        self.steps = steps
        self.initial_num_evals = initial_gradient_eval_steps
        self.max_num_evals = max_gradient_eval_steps
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.tensorboard = tensorboard
        self.constraint = constraint
        self._nqueries: Dict[int, int] = {}
        self.max_queries = max_queries
        self.mses = []
        self.queries = []
        self.rhos = []
        self.iter = -1
        assert constraint in ("l2", "linf")
        if constraint == "l2":
            self.distance = l2
        else:
            self.distance = linf

    def get_nqueries(self) -> Dict:
        return self._nqueries

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        self._nqueries = {i: 0 for i in range(len(originals))}
        criterion = get_criterion(criterion)
        self._criterion_is_adversarial = get_is_adversarial(criterion, model)
        self.mses.append([])
        self.queries.append([])
        self.rhos.append([])

        factor = self.args.layers
        n = originals.shape[2]
        z = np.array(range(0, 3))
        x = np.array(range(0, n, factor))
        y = np.array(range(0, n, factor))
        ma, mb, mc = np.meshgrid(z, x, y, indexing='ij')
        self.ma = torch.from_numpy(np.array(ma)).type(torch.LongTensor).cuda()
        self.mb = torch.from_numpy(np.array(mb)).type(torch.LongTensor).cuda()
        self.mc = torch.from_numpy(np.array(mc)).type(torch.LongTensor).cuda()
        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)
            x_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            x_advs = ep.astensor(starting_points)

        is_adv = self._is_adversarial(x_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        tb = TensorBoard(logdir=self.tensorboard)

        # Project the initialization to the boundary.
        x_advs = self._binary_search(self._is_adversarial, originals, x_advs)

        assert ep.all(self._is_adversarial(x_advs))

        distances = self.distance(originals, x_advs)

        for step in range(self.steps):
            delta = self.select_delta(originals, distances, step)

            # Choose number of gradient estimation steps.
            num_gradient_estimation_steps = int(
                min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals])
            )

            raw_adv = x_advs.raw.clone().detach()
            raw_adv.requires_grad = True
            Fcadv = model(raw_adv)
            Fcc_id = Fcadv.argmax(-1)
            Fcc = Fcadv[0][Fcc_id]
            Fcadv[0][Fcc_id] = -1
            F_else = Fcadv.max()
            Sx = Fcc - F_else
            Sx.backward()
            print("step: ", end='')
            print(step)

            '''print(raw_adv.grad.shape)
            print(type(raw_adv.grad))'''
            #p = raw_adv.grad[0, self.ma, self.mb, self.mc].detach().clone().unsqueeze(0).type(dtype)


            self.pra = False

            if self.pra:
                grad = raw_adv.grad.detach().clone()
                p = raw_adv.grad.detach().clone()*10
                grad_HR = F.interpolate(grad[0, self.ma, self.mb, self.mc].unsqueeze(0), scale_factor=self.args.layers,
                                       mode='bilinear', align_corners=True)
                grad_HR[0, self.ma, self.mb, self.mc] = grad[0, self.ma, self.mb, self.mc]
                #grad_random = torch.normal(mean=torch.zeros(1, 3, 224, 224)).type(dtype)
                grad_random = torch.rand(1, 3, 224, 224).type(dtype) * 2 - 1
                grad_random[0, self.ma, self.mb, self.mc] = grad[0, self.ma, self.mb, self.mc]
                #print(img_HR.shape)
                print("bilinear: ", end='')
                print(torch.cosine_similarity(grad.reshape(-1), grad_HR.reshape(-1), 0))
                print(torch.cosine_similarity(grad.reshape(-1), grad_random.reshape(-1), 0))
                #print(torch.mean(torch.abs(p)))


                #qwq = self.train(p).squeeze()


                #print(p)
                #print(qwq)
                #rho_all = torch.cosine_similarity(grad.reshape(-1), qwq.reshape(-1), 0)
                #print("rho_all: ", end='')
                #print(rho_all)

                print("下采样再上采样：")
                p = F.interpolate(grad, scale_factor=1./self.args.layers, mode="area")
                print(p.shape)
                pp = F.interpolate(p, scale_factor=self.args.layers, mode="bilinear", align_corners=False)
                pp2 = self.train(grad).squeeze()
                print(torch.cosine_similarity(grad.reshape(-1), pp.reshape(-1), 0))
                print(torch.cosine_similarity(grad.reshape(-1), pp2.reshape(-1), 0))

                from skimage import transform
                p = transform.resize(raw_adv.grad.squeeze().cpu().numpy().transpose(1, 2, 0), (n // 4, n // 4, 3)).transpose(2, 0, 1)
                p = transform.resize(p.transpose(1, 2, 0), (n, n, 3)).transpose(2, 0, 1)
                qwq = torch.from_numpy(p).type(dtype)
                qwq[self.ma, self.mb, self.mc] = grad[0, self.ma, self.mb, self.mc]
                print(torch.cosine_similarity(grad.reshape(-1), qwq.reshape(-1), 0))
                while True:
                    x = 1

            #print(qwq)
            #mmp = p-qwq
            #print(mmp[0][0])



            #print(qwq.shape)
            #print(raw_adv.grad.shape)


            gradients = self.approximate_gradients(
                self._is_adversarial, x_advs, num_gradient_estimation_steps, delta, originals
            )

            rho = torch.cosine_similarity(raw_adv.grad.view(-1), gradients.raw.view(-1), 0)
            #print(raw_adv.grad.shape)
            #print(gradients.raw.shape)
            #rho2 = torch.cosine_similarity(raw_adv.grad[0, self.ma, self.mb, self.mc].view(-1), gradients.raw[self.ma, self.mb, self.mc].view(-1), 0)
            print("rho: ", end='')
            print(rho.item())

            #print("rho2:")
            #print(rho2)
            self.rhos[self.iter].append(rho.item())

            if self.constraint == "linf":
                update = ep.sign(gradients)
            else:
                update = gradients

            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilons = distances / math.sqrt(step + 1)

                while True:
                    x_advs_proposals = ep.clip(
                        x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1
                    )
                    success = self._is_adversarial(x_advs_proposals)
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)

                    if ep.all(success):
                        break

                # Update the sample.
                x_advs = ep.clip(
                    x_advs + atleast_kd(epsilons, update.ndim) * update, 0, 1
                )

                assert ep.all(self._is_adversarial(x_advs))

                # Binary search to return to the boundary.
                x_advs = self._binary_search(self._is_adversarial, originals, x_advs)

                assert ep.all(self._is_adversarial(x_advs))

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons_grid = ep.expand_dims(
                    ep.from_numpy(
                        distances,
                        np.logspace(-4, 0, num=20, endpoint=True, dtype=np.float32),
                    ),
                    1,
                ) * ep.expand_dims(distances, 0)

                proposals_list = []

                for epsilons in epsilons_grid:
                    x_advs_proposals = (
                        x_advs + atleast_kd(epsilons, update.ndim) * update
                    )
                    x_advs_proposals = ep.clip(x_advs_proposals, 0, 1)

                    mask = self._is_adversarial(x_advs_proposals)

                    x_advs_proposals = self._binary_search(
                        self._is_adversarial, originals, x_advs_proposals
                    )

                    # only use new values where initial guess was already adversarial
                    x_advs_proposals = ep.where(
                        atleast_kd(mask, x_advs.ndim), x_advs_proposals, x_advs
                    )

                    proposals_list.append(x_advs_proposals)

                proposals = ep.stack(proposals_list, 0)
                proposals_distances = self.distance(
                    ep.expand_dims(originals, 0), proposals
                )
                minimal_idx = ep.argmin(proposals_distances, 0)

                x_advs = proposals[minimal_idx]

            distances = self.distance(originals, x_advs)

            # log stats
            mse = self.compute_mse(originals.raw, x_advs.raw)
            print("Adv MSE:", end=' ')
            print(mse)
            self.mses[self.iter].append(mse)
            # print(self._nqueries.values())
            self.queries[self.iter].append(self._nqueries[0])

            if all(v > self.max_queries for v in self._nqueries.values()):
                #print("Max queries attained for all the images.")
                break

        return restore_type(x_advs)

    def approximate_gradients(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        x_advs: ep.Tensor,
        steps: int,
        delta: ep.Tensor,
        originals,
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))
        if self.constraint == "l2":
            #rv = ep.normal(x_advs, noise_shape)
            #rv = self.gnn_generator(originals, x_advs, noise_shape)
            #self.practice(originals, x_advs, noise_shape)
            rv = self.deep_generator(originals, x_advs, noise_shape)
        elif self.constraint == "linf":
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)

        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12

        scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        #rv = (perturbed - x_advs) / 2
        rv = (perturbed - x_advs) / atleast_kd(ep.expand_dims(delta, 0), rv.ndim)




        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = is_adversarial(perturbed[step])
            multipliers_list.append(
                ep.where(
                    decision,
                    ep.ones(x_advs, (len(x_advs,))),
                    -ep.ones(x_advs, (len(decision,))),
                )
            )
        # (steps, bs, ...)
        multipliers = ep.stack(multipliers_list, 0)

        vals = ep.where(
            ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
            multipliers,
            multipliers - ep.mean(multipliers, axis=0, keepdims=True),
        )
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)

        #grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12

        #grad = ep.astensor(self.train(grad.raw.clone().detach()).squeeze())
        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
        return grad

    def _project(
        self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
    ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.
        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == "linf":
            perturbation = perturbed - originals

            # ep.clip does not support tensors as min/max
            clipped_perturbed = ep.where(
                perturbation > epsilons, originals + epsilons, perturbed
            )
            clipped_perturbed = ep.where(
                perturbation < -epsilons, originals - epsilons, clipped_perturbed
            )
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

    def _binary_search(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        originals: ep.Tensor,
        perturbed: ep.Tensor,
    ) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        d = np.prod(perturbed.shape[1:])
        if self.constraint == "linf":
            highs = linf(originals, perturbed)

            # TODO: Check if the threshold is correct
            #  empirically this seems to be too low
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = self.gamma / (d * math.sqrt(d))

        lows = ep.zeros_like(highs)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs

        while ep.any(highs - lows > thresholds):
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids

            if reached_numerical_precision:
                # TODO: warn user
                break

        res = self._project(originals, perturbed, highs)

        return res

    def select_delta(
        self, originals: ep.Tensor, distances: ep.Tensor, step: int
    ) -> ep.Tensor:
        result: ep.Tensor
        if step == 0:
            result = 0.1 * ep.ones_like(distances)
        else:
            d = np.prod(originals.shape[1:])

            if self.constraint == "linf":
                theta = self.gamma / (d * d)
                result = d * theta * distances
            else:
                theta = self.gamma / (d * np.sqrt(d))
                result = np.sqrt(d) * theta * distances

        return result

    def _is_adversarial(self, perturbed: ep.Tensor) -> ep.Tensor:
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1
        is_advs = self._criterion_is_adversarial(perturbed)
        return is_advs

    '''def gnn_generator(self, originals, x_advs, noise_shape):
        print("noise")
        print(noise_shape)
        while True:
            x=1
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            graph = self.image_graph[0]
            self.image_graph.reprocess(features=features, k=self.args.k)

            for j in range(noise_shape[0]):
                self.image_graph.set_labels()
                model = GCN(in_feats=graph.ndata["feat"].shape[1],
                            n_hidden=self.args.hidden,
                            n_classes=1,
                            n_layers=3,
                            dropout=self.args.dropout)
                p = train(model, graph, self.args).reshape(3, n, n)
                noise[j, i, :, :, :] = p.detach()
        return ep.astensor(noise)

    def graph_filter_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            graph = self.image_graph[0]
            #self.image_graph.reprocess(features=features)
            for j in range(noise_shape[0]):
                features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                conv = GraphConv(n*n*3, n*n*3, norm='both', weight=False, bias=False)
                alpha = self.args.alpha
                layers = self.args.layers
                p = (1-alpha) * conv(graph, features) + alpha * features
                for k in range(layers-1):
                    p = (1-alpha) * conv(graph, p) + alpha * features
                noise[j, i, :, :, :] = p.reshape(3, n, n)
        return ep.astensor(noise)

    def ep_graph_filter_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            self.image_graph.reprocess(originals[i].raw)
            graph = self.image_graph[0]
            for j in range(noise_shape[0]):
                features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                conv = GraphConv(n*n*3, n*n*3, norm='none', weight=False, bias=False)
                alpha = self.args.alpha
                layers = self.args.layers
                p = (1-alpha) * conv(graph, features, edge_weight=self.image_graph.graph.edata['h']) + alpha * features
                for k in range(layers-1):
                    p = (1-alpha) * conv(graph, p, edge_weight=self.image_graph.graph.edata['h']) + alpha * features
                #p = p * self.image_graph.mask
                noise[j, i, :, :, :] = p.reshape(3, n, n)
        return ep.astensor(noise)'''

    def deep_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        bf = self.args.bf
        for i in range(N):
            if bf:
                ori = originals[i].raw.cpu().numpy()
                ori = np.moveaxis(ori, 0, 2)

            #ori = np.moveaxis(ori, 0, 2)
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                #features = np.random.normal(0., 1., (1, 3, n, n)).astype(np.float32)
                #features = torch.from_numpy(features).cuda()
                #p = self.train(ori.clone().detach(), features).squeeze()
                #p = torch.zeros(1, 3, n, n).type(torch.FloatTensor).cuda()
                '''if bf:
                    features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                    p = cv2.ximgproc.jointBilateralFilter(ori, features,
                                                          16, 16 / 255 * 2, 16)
                    p = torch.from_numpy(np.moveaxis(p, 2, 0)).unsqueeze(0).type(dtype)
                else:'''
                p = torch.normal(mean=torch.zeros(1, 3, n, n)).type(dtype)
                #p[0, self.ma, self.mb, self.mc] = torch.rand(1, 3, self.ma.shape[1], self.ma.shape[2]).type(dtype) * 2. - 1.
                #grad = ep.astensor(self.train(grad.raw.clone().detach()).squeeze())
                ret = self.train(p).squeeze()   #(3, n, n)
                if bf:
                    ret = cv2.ximgproc.jointBilateralFilter(ori, np.moveaxis(ret.cpu().numpy(), 0, 2), 16, 16/255*2, 16)
                #print("shape of p: ", end="")
                #print(p.shape)

                """layers / 255 * 2"""
                if bf:
                    noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(ret, 2, 0)).cuda()
                else:
                    noise[j, i, :, :, :] = ret
        return ep.astensor(noise)

    def compute_mse(self, x1, x2):
        dis = np.linalg.norm(x1.cpu().numpy() - x2.cpu().numpy())
        mse = dis ** 2 / np.prod(x1.cpu().numpy().shape)
        return mse

    def train(self, image):
        factor = self.args.layers
        if factor == 4:
            num_iter = 100
            reg_noise_std = 0.03
        elif factor == 8:
            num_iter = 100
            reg_noise_std = 0.05
        NET_TYPE = 'skip'  # UNet, ResNet
        #n = image.shape[2]
        if self.args.bf:
            img = F.interpolate(image, scale_factor=1./self.args.layers, mode="area", align_corners=None)
            #print(img.shape)
        else:
            img = image[0, self.ma, self.mb, self.mc].unsqueeze(0)
        #img = img.clip(-1, 1)
        #img_saved = img.clone().detach()
        img_saved = image[0, self.ma, self.mb, self.mc].unsqueeze(0)
        add = 0.5

        img_HR = F.interpolate(img, scale_factor=self.args.layers, mode='bilinear', align_corners=False).detach()
        img_HR[0, self.ma, self.mb, self.mc] = img.detach().clone()

        img = (img + 1.) / 2
        img_HR = (img_HR + 1.) / 2
        downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5,
                                  preserve_size=True).type(dtype)

        net = self.get_net(input_depth, 'skip', pad,
                      skip_n33d=128,
                      skip_n33u=128,
                      skip_n11=4,
                      num_scales=5,
                      upsample_mode='bilinear',
                      downsample_mode='lanczos2').type(dtype) #lanczos2
        net_input = self.get_noise(input_depth, INPUT, (image.shape[2], image.shape[3])).type(dtype).detach()
        net_input_saved = net_input.clone().detach()
        noise = net_input.detach().clone()
        #ret = torch.normal(mean=torch.zeros(image.shape)).type(dtype)
        #print(ret.shape)
        parameters = self.get_params(OPT_OVER, net, net_input)
        optimizer = torch.optim.Adam(parameters, lr=self.args.lr)

        for j in range(num_iter):
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            optimizer.zero_grad()
            out_HR = net(net_input)
            out_LR = downsampler(out_HR)
            loss_H = 0.#mse(out_HR, img_HR)#-1. * cs(out_HR.view(-1), img_HR.view(-1))
            total_loss = mse(out_LR, img)# - 0.0 * cs(out_LR.view(-1), img.view(-1))
            #total_loss = -1.0 * cs(out_LR.view(-1), img.view(-1))
            total_loss = (total_loss + loss_H)# / 2
            loss_var = torch.var(out_HR)
            a = 0.9
            #print(loss_var)
            total_loss = a * total_loss + (1. - a) * loss_var
            total_loss.backward()
            optimizer.step()
            #if j == num_iter - 1:
            if self.pra:
                print(j, end=" : ")
                #print(mse((out_LR.view(-1)*2-1.), img_saved.view(-1)))
                rho_all = torch.cosine_similarity(image.reshape(-1), out_HR.reshape(-1)*2-1., 0)
                #print(total_loss.item())
                print(rho_all.item())
                #print(out_HR.detach().mean().item())
        out_HR = net(net_input) * 2 - 1.
        #out_HR[0, self.ma, self.mb, self.mc] = img_saved
        return out_HR.detach()

    '''def practice(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            factor = self.args.layers
            image = originals[i].raw
            if factor == 4:
                num_iter = 8000
                reg_noise_std = 0.03
            elif factor == 8:
                num_iter = 8000
                reg_noise_std = 0.05
            NET_TYPE = 'skip'  # UNet, ResNet

            downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5,
                                      preserve_size=True).type(dtype)
            net = self.get_net(input_depth, 'ResNet', pad,
                               skip_n33d=128,
                               skip_n33u=128,
                               skip_n11=4,
                               num_scales=5,
                               upsample_mode='bilinear').type(dtype)

            parameters = self.get_params(OPT_OVER, net)
            optimizer = torch.optim.Adam(parameters, lr=self.args.lr)

            for j in range(num_iter):

                net_input = self.get_noise(input_depth, INPUT, (image.shape[1], image.shape[2])).type(dtype).detach()
                img = torch.normal(mean=torch.zeros(3, self.ma.shape[1], self.ma.shape[2])).cuda()
                #noise = net_input.detach().clone()
                #if reg_noise_std > 0:
                    #net_input = net_input + (noise.normal_() * reg_noise_std)
                optimizer.zero_grad()
                total_loss = 0
                for k in range(48):
                    out_HR = net(net_input)
                    out_LR = downsampler(out_HR)
                    total_loss += -1.0 * mse(out_LR.view(-1), img.view(-1))
                total_loss /= 48.
                total_loss.backward()
                optimizer.step()
                # if j == num_iter - 1:
                print(j, end=" : ")
                print(total_loss.item())
            #out_HR = net(net_input)
            #return out_HR.detach()
            return 0


            #ori = np.moveaxis(ori, 0, 2)
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                #features = np.random.normal(0., 1., (1, 3, n, n)).astype(np.float32)
                #features = torch.from_numpy(features).cuda()
                #p = self.train(ori.clone().detach(), features).squeeze()
                p = torch.zeros(1, 3, n, n).type(torch.FloatTensor).cuda()
                p[0, self.ma, self.mb, self.mc] = torch.normal(mean=torch.zeros(1, 3, self.ma.shape[1], self.ma.shape[2])).cuda()
                #grad = ep.astensor(self.train(grad.raw.clone().detach()).squeeze())
                ret = self.train(p).squeeze()
                #print("shape of p: ", end="")
                #print(p.shape)

                """layers / 255 * 2"""
                noise[j, i, :, :, :] = ret

        return ep.astensor(noise)'''

    def get_noise(self, input_depth, method, spatial_size, noise_type='n', var=1. / 10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)

            self.fill_noise(net_input, noise_type)
            net_input *= var
        elif method == 'meshgrid':
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                               np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
            meshgrid = np.concatenate([X[None, :], Y[None, :]])
            net_input = self.np_to_torch(meshgrid)
        else:
            assert False

        return net_input

    def fill_noise(self, x, noise_type):
        """Fills tensor `x` with noise of type `noise_type`."""
        if noise_type == 'u':
            x.uniform_()
        elif noise_type == 'n':
            x.normal_()
        else:
            assert False
    def np_to_torch(self, img_np):
        '''Converts image in numpy.array to torch.Tensor.

        From C x W x H [0..1] to  C x W x H [0..1]
        '''
        return torch.from_numpy(img_np)[None, :]
    def get_params(self, opt_over, net, net_input=None, downsampler=None):
        '''Returns parameters that we want to optimize over.
        Args:
            opt_over: comma separated list, e.g. "net,input" or "net"
            net: network
            net_input: torch.Tensor that stores input `z`
        '''
        opt_over_list = opt_over.split(',')
        params = []

        for opt in opt_over_list:

            if opt == 'net':
                params += [x for x in net.parameters()]
            elif opt == 'down':
                assert downsampler is not None
                params = [x for x in downsampler.parameters()]
            elif opt == 'input':
                net_input.requires_grad = True
                params += [net_input]
            else:
                assert False, 'what is it?'

        return params

    def get_net(self, input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128,
                skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
        if NET_TYPE == 'ResNet':
            # TODO
            #net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
            net = ResNet(input_depth, 3, 8, 32, need_sigmoid=True, act_fun='LeakyReLU')
        elif NET_TYPE == 'skip':
            net = skip(input_depth, n_channels,
                       num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
                       num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
                       num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
                       upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                       need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

        elif NET_TYPE == 'texture_nets':
            net = get_texture_nets(inp=input_depth, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False, pad=pad)

        elif NET_TYPE == 'UNet':
            net = UNet(num_input_channels=input_depth, num_output_channels=3,
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True,
                       need_bias=True)
        elif NET_TYPE == 'identity':
            assert input_depth == 3
            net = nn.Sequential()
        else:
            assert False

        return net

class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None,
                 preserve_size=False):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'

        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:

            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)

def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val


    else:
        assert False, 'wrong method name'

    kernel /= kernel.sum()

    return kernel
