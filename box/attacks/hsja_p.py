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
from skimage import transform
from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from ..distances import l2, linf
from typing import Callable, Union, Optional, Tuple, List, Any, Dict
import cv2
import torch.nn.functional as F

class HSJA_P(MinimizationAttack):
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
        self.steps = 5000
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
        self.confirms = []
        self.iter = -1
        self.save_method = self.args.method
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
        self.confirms.append([])
        self.args.method = self.save_method
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
            self.starting_points = starting_points
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

        self.last_size = 5
        self.last = torch.zeros(self.last_size, x_advs.shape[1], x_advs.shape[2], x_advs.shape[3]).type(torch.cuda.FloatTensor)
        self.grads = torch.zeros(self.last_size, x_advs.shape[1], x_advs.shape[2], x_advs.shape[3]).type(torch.cuda.FloatTensor)
        self.last_iter = None
        #self.last_adv = None
        #self.advd = None
        self.last_true_grad = None
        self.start = False
        if self.args.method == 'RBD':
            self.rbd = self.calc_rbd(originals)
        for step in range(self.steps):
            delta = self.select_delta(originals, distances, step)

            # Choose number of gradient estimation steps.
            num_gradient_estimation_steps = int(
                min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals])
            )

            '''raw_adv = x_advs.raw.clone().detach()
            raw_adv.requires_grad = True
            Fcadv = model(raw_adv)
            Fcc_id = Fcadv.argmax(-1)
            Fcc = Fcadv[0][Fcc_id]
            Fcadv[0][Fcc_id] = -1
            F_else = Fcadv.max()
            Sx = Fcc - F_else
            Sx.backward()'''
            #a = x_advs - originals  # my push vector
            '''print("a: ", end='')
            print(a.shape)'''
            #a /= ep.norms.l2(atleast_kd(flatten(a), a.ndim)) + 1e-12

            if self.last_true_grad is not None:
                pass
                #print(self.advd, end=":::")
                #print("grad: ", end='')
                #print("step ", end='')
                #print(step, end=': ')
                #sim = torch.cosine_similarity(raw_adv.grad.view(-1), self.last_true_grad.view(-1), 0).item()
                #print("True:", end=' ')
                #print(sim)
                #self.confirms[self.iter].append(sim)
            #self.last_true_grad = raw_adv.grad
            '''raw_ori = originals.raw.clone().detach()
            raw_ori.requires_grad = True
            Fcadv = model(raw_ori)
            Fcc_id = Fcadv.argmax(-1)
            Fcc = Fcadv[0][Fcc_id]
            Fcadv[0][Fcc_id] = -1
            F_else = Fcadv.max()
            Sx = Fcc - F_else
            Sx.backward()
            print(torch.cosine_similarity(raw_adv.grad.view(-1), raw_ori.grad.view(-1), 0))'''
            gradients = self.approximate_gradients(
                self._is_adversarial, x_advs, num_gradient_estimation_steps, delta, originals, step
            )
            #gradients = ep.astensor(raw_adv.grad)
            #rho = torch.cosine_similarity(raw_adv.grad.view(-1), gradients.raw.view(-1), 0)
            #rho = torch.cosine_similarity(raw_adv.grad.reshape(-1), gradients.raw.reshape(-1), 0)
            #print("RHO: ", end='')
            #print(rho.item())

            #self.rhos[self.iter].append(rho.item())
            self.rhos[self.iter].append(0)
            if self.constraint == "linf":
                update = ep.sign(gradients)
            else:
                update = gradients
            #print(ep.abs(update).mean() * distances)
            '''epsilons = distances# / math.sqrt(step + 1)

            for i in range(50):
                x_advs_proposals = ep.clip(
                    originals + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1
                )
                success = self._is_adversarial(x_advs_proposals)
                if ep.all(success):
                    break
                epsilons = ep.where(success, epsilons, epsilons / 2.0)'''
            #x_advs_proposals = ep.clip(originals + atleast_kd(distances, originals.ndim) * update, 0, 1)
            #print(originals)
            #print(x_advs_proposals)

            '''success = self._is_adversarial(x_advs_proposals)
            #print(success)
            if not ep.all(success):
                continue
            x_advs = self._binary_search(self._is_adversarial, originals, x_advs_proposals)'''

            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilons = distances / math.sqrt(step + 1)

                while True:
                    x_advs_proposals = ep.clip(
                        x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1
                    )
                    '''print("########################")
                    print(atleast_kd(epsilons, x_advs.ndim))
                    print(x_advs)
                    print("########################")'''
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
            print(2)
            distances = self.distance(originals, x_advs)
            '''if self.last_adv is not None:
                self.advd = self.distance(self.last_adv, x_advs).item()
            self.last_adv = x_advs'''
            # log stats
            mse = self.compute_mse(originals.raw, x_advs.raw)
            print("MSE: ", end='')
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
        num_step,
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))
        if self.constraint == "l2":
            if self.args.method == 'HSJA':
                rv = ep.normal(x_advs, noise_shape)
            elif self.args.method == 'QEBA':
                rv = self.resize_generator(originals, x_advs, noise_shape)
            elif self.args.method == 'JBF':
                rv = self.dba_generator(originals, x_advs, noise_shape)
            elif self.args.method == 'GUIDE':
                rv = self.guide_generator(originals, x_advs, noise_shape)
            elif self.args.method == 'GAUSS':
                rv = self.gauss_generator(originals, x_advs, noise_shape)
            elif self.args.method == 'SWF':
                self.swf = SideWindowFilter(radius=self.args.r, iteration=self.args.iteration)
                rv = self.swf_generator(originals, x_advs, noise_shape)
            elif self.args.method == 'RBD':
                rv = self.rbd_generator(originals, x_advs, noise_shape)
            else:
                raise False
        elif self.constraint == "linf":
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12
        scaled_rv = atleast_kd(delta, rv.ndim) * rv


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
        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
        ##########################################
        if self.args.time:
            val_grads: List[ep.Tensor] = []
            pre = 0
            if self.last_iter is None:
                self.last_iter = 0
            else:
                for i in range(self.last_size):
                    cos = torch.cosine_similarity(self.grads[i].view(-1), grad.raw.view(-1), 0).item()
                    if self.args.method == 'HSJA':
                        q = 0.001 #0.001, 正在调参MNIST 调为0.003
                    else:
                        q = 0.005
                    # inception : q=0.002, 0.01, dis < 0.3, last_size=10
                    # 4.4 下午17:57，调参为d<0.2， 之后densenet先调回0.3
                    if self.distance(self.last[i], x_advs) < 0.15 and cos > q:  #<0.2

                    #if num_step > 120:
                        #print(cos)
                        val_grads.append(ep.astensor(self.grads[i].unsqueeze(0)))
                        pre = pre + 1

                self.last_iter = (self.last_iter + 1) % self.last_size
            # JBF:1.85e-5
            if self.args.method == 'HSJA':
                th = 0
            else:
                th = 0
            if pre > th:
                print("PRE: ", end='')
                print(pre)
                #val_grads.append(grad)
                #grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad /= ep.norms.l2(atleast_kd(flatten(pre_grad), pre_grad.ndim)) + 1e-12
                grad = 2*grad - pre_grad
                grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
            self.last[self.last_iter] = x_advs.raw
            self.grads[self.last_iter] = grad.raw
            # JBF:1.85e-5
            if self.args.method == 'HSJA':
                th = 0
            else:
                th = 0
            if pre > th:
                print("PRE: ", end='')
                print(pre)
                #val_grads.append(grad)
                #grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad /= ep.norms.l2(atleast_kd(flatten(pre_grad), pre_grad.ndim)) + 1e-12
                grad = 2*grad - pre_grad
                grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
            self.last[self.last_iter] = x_advs.raw
            self.grads[self.last_iter] = grad.raw

        elif self.args.ntime:
            val_grads: List[ep.Tensor] = []
            val_grads.append(grad)
            pre = 0
            if self.last_iter is None:
                self.last_iter = 0
            else:
                for i in range(self.last_size):
                    # inception : q=0.002, 0.01, dis < 0.3, last_size=10
                    # 4.4 下午17:57，调参为d<0.2， 之后densenet先调回0.3
                    if self.distance(self.last[i], x_advs) < 0.2:
                        # if num_step > 120:
                        # print(cos)
                        val_grads.append(ep.astensor(self.grads[i].unsqueeze(0)))
                        pre = pre + 1

                self.last_iter = (self.last_iter + 1) % self.last_size
            # JBF:1.85e-5
            if self.args.method == 'HSJA':
                th = 0
            else:
                th = 0
            if pre > th:
                print("PRE: ", end='')
                print(pre)
                # val_grads.append(grad)
                # grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad = ep.mean(ep.stack(val_grads, 0), 0)
                pre_grad /= ep.norms.l2(atleast_kd(flatten(pre_grad), pre_grad.ndim)) + 1e-12
                grad = pre_grad  #2 * grad - pre_grad
                grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
            self.last[self.last_iter] = x_advs.raw
            self.grads[self.last_iter] = grad.raw

        '''alpha = 0.8
        # print(self.advd)
        if self.advd is not None:
            if self.advd < 0.4:
                grad = alpha * grad + (1 - alpha) * self.last
                # print(1)
        self.last = grad'''

        #grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12

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

    def compute_mse(self, x1, x2):
        dis = np.linalg.norm(x1.cpu().numpy() - x2.cpu().numpy())
        mse = dis ** 2 / np.prod(x1.cpu().numpy().shape)
        return mse

    def resize_generator(self, originals, x_advs, noise_shape):
        factor = 4
        N = originals.shape[0]
        # noise_shape: (steps, num_images, 3, 224, 224)
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        if noise_shape[2] == 3:
            for i in range(N):
                for j in range(noise_shape[0]):
                    shape = x_advs[i].shape
                    assert len(shape) == 3 and shape[0] == 3
                    p_small = torch.normal(mean=torch.zeros(1, shape[0], shape[1] // factor, shape[2] // factor).type(torch.cuda.FloatTensor))
                    #np.random.randn(shape[0], int(shape[1] / factor), int(shape[2] / factor))
                    # if (_ == 0):
                    #    print (p_small.shape)
                    #p = transform.resize(p_small.transpose(1, 2, 0), (n, n, 3)).transpose(2, 0, 1)
                    p = F.interpolate(p_small, scale_factor=4, mode='bilinear', align_corners=False)
                    noise[j, i, :, :, :] = p
        else:
            for i in range(N):
                for j in range(noise_shape[0]):
                    shape = x_advs[i][0].shape
                    assert len(shape) == 2
                    if self.args.f2:
                        factor = 2.
                    p_small = np.random.randn(int(shape[0] / factor), int(shape[1] / factor))
                    # if (_ == 0):
                    #    print (p_small.shape)
                    p = transform.resize(p_small, (n, n))
                    noise[j, i, :, :] = torch.from_numpy(p).to(torch.float32).cuda()
        return ep.astensor(noise)
    '''def resize_generator(self, originals, x_advs, noise_shape):
        factor = 4
        N = originals.shape[0]
        # noise_shape: (steps, num_images, 3, 224, 224)
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            for j in range(noise_shape[0]):
                shape = x_advs[i].shape
                assert len(shape) == 3 and shape[0] == 3
                p_small = np.random.randn(shape[0], int(shape[1] / factor), int(shape[2] / factor))
                # if (_ == 0):
                #    print (p_small.shape)
                p = transform.resize(p_small.transpose(1, 2, 0), (n, n, 3)).transpose(2, 0, 1)
                noise[j, i, :, :, :] = torch.from_numpy(p).to(torch.float32).cuda()
        return ep.astensor(noise)'''

    def jbf_generator(self, originals, x_advs, noise_shape):
        # noise_sh0ape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            ori = originals[i].raw.cpu().numpy()
            #ori = x_advs[i].raw.cpu().numpy()
            ori = np.moveaxis(ori, 0, 2)
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                layers = self.args.layers
                p = cv2.ximgproc.jointBilateralFilter(ori, features, 8, 32/255, 8)
                """layers / 255 * 2"""
                noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()

        return ep.astensor(noise)

    def dba_generator(self, originals, x_advs, noise_shape):
        # noise_sh0ape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #ori = originals[i].raw.cpu().numpy()
            ori = x_advs[i].raw.cpu().numpy()
            if noise_shape[2] == 3:
                ori = np.moveaxis(ori, 0, 2)
                for j in range(noise_shape[0]):
                    features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                    factor = 16
                    p = cv2.ximgproc.jointBilateralFilter(ori, features, factor*2, factor*2/255, factor/2)
                    noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()
            else:
                for j in range(noise_shape[0]):
                    features = np.random.normal(0., 1., (n, n)).astype(np.float32)
                    factor = 2
                    p = cv2.ximgproc.jointBilateralFilter(ori[0], features, factor*2, factor*4 / 255, factor/2)
                    noise[j, i, :, :] = torch.from_numpy(p).cuda()

        return ep.astensor(noise)

    def calc_rbd(self, originals):
        stp = (np.moveaxis(self.starting_points[0].cpu().numpy(), 0, 2) * 255).astype('uint8')
        import pyimgsaliency as psal
        rbd_adv = psal.get_saliency_rbd(stp) / 255
        ori = (np.moveaxis(originals[0].raw.cpu().numpy(), 0, 2) * 255).astype('uint8')
        rbd_ori = psal.get_saliency_rbd(ori) / 255
        # ori = x_advs[i].raw.cpu().numpy()
        # ori = np.moveaxis(ori, 0, 2)
        # mse1 = self.compute_mse(self.starting_points[i], x_advs[i].raw)
        # mse2 = self.compute_mse(originals[i].raw, x_advs[i].raw)
        # rate1 = mse1 / (mse2+mse1)
        # rate2 = mse2 / (mse2+mse1)
        # print(rate2)
        # rbd = rate2*rbd_adv + rate1*rbd_ori
        rbd_qwq = 0.5 * rbd_adv + 0.5 * rbd_ori
        rbd_qwq = torch.from_numpy(rbd_qwq).cuda()
        # rbd = torch.where(rbd_qwq > 0.3, rbd_qwq, 0.0*torch.ones_like(rbd_qwq))
        return rbd_qwq

    def rbd_generator(self, originals, x_advs, noise_shape):
        # noise_sh0ape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            '''stp = (np.moveaxis(self.starting_points[i].cpu().numpy(), 0, 2)*255).astype('uint8')
            rbd_adv = psal.get_saliency_rbd(stp) / 255
            ori = (np.moveaxis(originals[i].raw.cpu().numpy(), 0, 2) * 255).astype('uint8')
            rbd_ori = psal.get_saliency_rbd(ori) / 255
            #ori = x_advs[i].raw.cpu().numpy()
            #ori = np.moveaxis(ori, 0, 2)
            #mse1 = self.compute_mse(self.starting_points[i], x_advs[i].raw)
            #mse2 = self.compute_mse(originals[i].raw, x_advs[i].raw)
            #rate1 = mse1 / (mse2+mse1)
            #rate2 = mse2 / (mse2+mse1)
            #print(rate2)
            #rbd = rate2*rbd_adv + rate1*rbd_ori
            rbd_qwq = 0.5 * rbd_adv + 0.5 * rbd_ori
            rbd_qwq = torch.from_numpy(rbd_qwq).cuda()
            #rbd = torch.where(rbd_qwq > 0.3, rbd_qwq, 0.0*torch.ones_like(rbd_qwq))
            rbd = rbd_qwq'''
            rbd = self.rbd
            qwq = torch.normal(mean=torch.zeros(noise_shape[0], 3, n, n)).cuda()
            qwq = qwq * rbd
            factor = 4
            aaa = F.interpolate(qwq, scale_factor=1. / factor, mode='area')
            noise[:, i, :, :, :] = F.interpolate(aaa, scale_factor=factor, mode='bilinear', align_corners=False)
            '''
            for j in range(noise_shape[0]):
                #features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                #p = cv2.ximgproc.jointBilateralFilter(ori, features, 8, 32/255, 8)
                #qwq = torch.from_numpy(np.random.normal(0., 1., (1, 3, n, n)).astype(np.float32) * rbd).cuda()
                qwq = torch.normal(mean=torch.zeros(1, 3, n, n)).cuda()
                qwq = qwq * rbd
                factor = 4
                aaa = F.interpolate(qwq, scale_factor=1./factor, mode='area')
                #print(aaa.shape)
                noise[j, i, :, :, :] = F.interpolate(aaa, scale_factor=factor, mode='bilinear', align_corners=False)'''
        return ep.astensor(noise)

    def guide_generator(self, originals, x_advs, noise_shape):
        # noise_sh0ape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            ori = originals[i].raw.cpu().numpy()
            #ori = x_advs[i].raw.cpu().numpy()
            ori = np.moveaxis(ori, 0, 2)
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                layers = self.args.layers
                p = cv2.ximgproc.jointBilateralFilter(ori, features, 8, 32/255, 8)
                p = cv2.ximgproc.guidedFilter(ori, features, 8, math.sqrt(8) / 255)
                """layers / 255 * 2"""
                noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()

        return ep.astensor(noise)
    def gauss_generator(self, originals, x_advs, noise_shape):
        # noise_sh0ape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            ori = originals[i].raw.cpu().numpy()
            #ori = x_advs[i].raw.cpu().numpy()
            ori = np.moveaxis(ori, 0, 2)
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                layers = self.args.layers
                p = cv2.GaussianBlur(features, (17, 17), 8)
                """layers / 255 * 2"""
                noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()

        return ep.astensor(noise)

    def swf_generator(self, originals, x_advs, noise_shape):
        # noise_shape: (steps, num_images, 3, 224, 224)
        N = originals.shape[0]
        n = noise_shape[3]
        noise = torch.zeros(noise_shape)
        if torch.cuda.is_available():
            noise = noise.cuda()
        for i in range(N):
            #features = torch.cat((originals[i].raw.reshape(-1, 1), x_advs[i].raw.reshape(-1, 1)), -1)
            for j in range(noise_shape[0]):
                #features = torch.from_numpy(np.random.normal(0., 1., (n * n * 3, 1))).to(torch.float32).cuda()
                #features = np.random.normal(0., 1., (n, n, 3)).astype(np.float32)
                features = torch.from_numpy(np.random.normal(0., 1., originals[i].raw.shape).astype(np.float32)).cuda()
                #p = cv2.ximgproc.jointBilateralFiltokoker(np.moveaxis(originals[i].raw.cpu().numpy(), 0, 2), features,
                                                      #layers, layers/255*2, layers)
                p = self.swf(originals[i].raw.unsqueeze(0), features.unsqueeze(0)).squeeze()
                """layers / 255 * 2"""
                #noise[j, i, :, :, :] = torch.from_numpy(np.moveaxis(p, 2, 0)).cuda()
                noise[j, i, :, :, :] = p

        return ep.astensor(noise)



import torch
import torch.nn as nn
import torch.nn.functional as F


class SideWindowFilter(nn.Module):

    def __init__(self, radius, iteration, filter='box'):
        super(SideWindowFilter, self).__init__()
        self.radius = radius
        self.iteration = iteration
        self.kernel_size = 2 * self.radius + 1
        self.filter = filter

    def forward(self, im, pert):
        b, c, h, w = im.size()

        d = torch.zeros(b, 8, h, w, dtype=torch.float).cuda()
        dd = torch.zeros(b, 8, h, w, dtype=torch.float).cuda()
        res = im.clone()

        if self.filter.lower() == 'box':
            filter = torch.ones(1, 1, self.kernel_size, self.kernel_size).cuda()
            L, R, U, D = [filter.clone() for _ in range(4)]

            L[:, :, :, self.radius + 1:] = 0
            R[:, :, :, 0: self.radius] = 0
            U[:, :, self.radius + 1:, :] = 0
            D[:, :, 0: self.radius, :] = 0

            NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

            L, R, U, D = L / ((self.radius + 1) * self.kernel_size), R / ((self.radius + 1) * self.kernel_size), \
                         U / ((self.radius + 1) * self.kernel_size), D / ((self.radius + 1) * self.kernel_size)

            NW[:, :, :, self.radius + 1:] = 0
            NE[:, :, :, 0: self.radius] = 0
            SW[:, :, :, self.radius + 1:] = 0
            SE[:, :, :, 0: self.radius] = 0

            NW, NE, SW, SE = NW / ((self.radius + 1) ** 2), NE / ((self.radius + 1) ** 2), \
                             SW / ((self.radius + 1) ** 2), SE / ((self.radius + 1) ** 2)

            # sum = self.kernel_size * self.kernel_size
            # sum_L, sum_R, sum_U, sum_D, sum_NW, sum_NE, sum_SW, sum_SE = \
            #     (self.radius + 1) * self.kernel_size, (self.radius + 1) * self.kernel_size, \
            #     (self.radius + 1) * self.kernel_size, (self.radius + 1) * self.kernel_size, \
            #     (self.radius + 1) ** 2, (self.radius + 1) ** 2, (self.radius + 1) ** 2, (self.radius + 1) ** 2


        for ch in range(c):
            im_ch = im[:, ch, ::].clone().view(b, 1, h, w)
            pert_ch = pert[:, ch, ::].clone().view(b, 1, h, w)
            # print('im size in each channel:', im_ch.size())

            for i in range(self.iteration):
                # print('###', (F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) / sum_L -
                # im_ch).size(), d[:, 0,::].size())
                d[:, 0, ::] = F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) - im_ch
                d[:, 1, ::] = F.conv2d(input=im_ch, weight=R, padding=(self.radius, self.radius)) - im_ch
                d[:, 2, ::] = F.conv2d(input=im_ch, weight=U, padding=(self.radius, self.radius)) - im_ch
                d[:, 3, ::] = F.conv2d(input=im_ch, weight=D, padding=(self.radius, self.radius)) - im_ch
                d[:, 4, ::] = F.conv2d(input=im_ch, weight=NW, padding=(self.radius, self.radius)) - im_ch
                d[:, 5, ::] = F.conv2d(input=im_ch, weight=NE, padding=(self.radius, self.radius)) - im_ch
                d[:, 6, ::] = F.conv2d(input=im_ch, weight=SW, padding=(self.radius, self.radius)) - im_ch
                d[:, 7, ::] = F.conv2d(input=im_ch, weight=SE, padding=(self.radius, self.radius)) - im_ch

                dd[:, 0, ::] = F.conv2d(input=pert_ch, weight=L, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 1, ::] = F.conv2d(input=pert_ch, weight=R, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 2, ::] = F.conv2d(input=pert_ch, weight=U, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 3, ::] = F.conv2d(input=pert_ch, weight=D, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 4, ::] = F.conv2d(input=pert_ch, weight=NW, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 5, ::] = F.conv2d(input=pert_ch, weight=NE, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 6, ::] = F.conv2d(input=pert_ch, weight=SW, padding=(self.radius, self.radius)) - pert_ch
                dd[:, 7, ::] = F.conv2d(input=pert_ch, weight=SE, padding=(self.radius, self.radius)) - pert_ch

                d_abs = torch.abs(d)

                mask_min = torch.argmin(d_abs, dim=1, keepdim=True)

                dm = torch.gather(input=d, dim=1, index=mask_min)
                ddm = torch.gather(input=dd, dim=1, index=mask_min)
                im_ch = dm + im_ch
                pert_ch = ddm + pert_ch

            res[:, ch, ::] = pert_ch
        return res



'''        
        if self.alpha_a > 0.1:
            while True:
                #sr = ep.normal(x_advs, x_advs.shape)s
                sr = ep.astensor(torch.normal(mean=-1. * scaled_a.raw, std=0.2).type(torch.cuda.FloatTensor))
                sr /= ep.norms.l2(atleast_kd(flatten(sr), sr.ndim)) + 1e-12
                sr = atleast_kd(delta, sr.ndim) * sr
                # print("shape")
                # print(sr.shape)
                if num >= 50:
                    self.alpha_a *= 0.7
                    num = 0
                perturbed = x_advs + scaled_a * self.alpha_a + sr
                perturbed = ep.clip(perturbed, 0, 1)
                if is_adversarial(perturbed):
                    ar = perturbed - x_advs
                    r = ar - scaled_a * self.alpha_a
                    vals.append(r)
                    num_r += 1
                    num = 0
                    if num_r == 1:
                        break
                else:
                    num_reject += 1
                    num += 1'''

'''while True:
            sr = ep.normal(x_advs, x_advs.shape)
            #sr = ep.normal(x_advs, (1, 3, x_advs.shape[2], x_advs.shape[3]))
            #sr = F.interpolate(sr, scale_factor=self.args.layers, mode='bilinear', align_corners=False)
            sr /= ep.norms.l2(atleast_kd(flatten(sr), sr.ndim)) + 1e-12
            sr = atleast_kd(delta, sr.ndim) * sr
            #print("shape")
            #print(sr.shape)
            if num >= 10:
                self.alpha_a *= 0.5
                num = 0
            perturbed = x_advs + scaled_a * self.alpha_a + sr
            perturbed = ep.clip(perturbed, 0, 1)
            if is_adversarial(perturbed):
                ar = perturbed - x_advs
                r = ar - scaled_a * self.alpha_a
                vals.append(r)
                num_r += 1
                #if num == 0:
                    #self.alpha_a /= 0.5
                num = 0
                if num_r == 1:
                    break
            else:
                perturbed = x_advs + scaled_a * self.alpha_a - sr
                perturbed = ep.clip(perturbed, 0, 1)
                if is_adversarial(perturbed):
                    ar = perturbed - x_advs
                    r = ar - scaled_a * self.alpha_a
                    vals.append(r)
                    num_r += 1
                    #if num == 0:
                        #self.alpha_a /= 0.5
                    num = 0
                    if num_r == 1:
                        break
                else:
                    num_reject += 1
                    num += 1'''

'''
            num = 0
            for sr in scaled_rv:
                if num >= 10:
                    self.alpha_a *= 0.8
                    num = 0
                #print("sr: ", end='')
                #sr = sr.reshape((1,) + sr.shape)
                perturbed = x_advs + scaled_a*self.alpha_a + sr
                perturbed = ep.clip(perturbed, 0, 1)
                #print("perturbed: ", end='')
                #print(perturbed.shape)
                if is_adversarial(perturbed):
                    ar = perturbed - x_advs
                    r = ar - scaled_a*self.alpha_a
                    vals.append(r)
                    num_r += 1
                    num = 0
                    if num_r == 5:
                        ok = True
                        break
                else:
                    perturbed = x_advs + scaled_a*self.alpha_a - sr
                    perturbed = ep.clip(perturbed, 0, 1)
                    if is_adversarial(perturbed):
                        ar = perturbed - x_advs
                        r = ar - scaled_a*self.alpha_a
                        vals.append(r)
                        num_r += 1
                        num = 0
                        if num_r == 5:
                            ok = True
                            break
                    else:
                        num_reject += 1
                        num += 1
            if ok:
                break
            if num_reject == steps:
                self.alpha_a *= 0.8
                num_reject = 0
            else:
                break'''
'''vals: List[ep.Tensor] = []
#print("a: ", end='')
#print(scaled_a.shape)
#perturbed = x_advs + scaled_a
#perturbed = ep.clip(perturbed, 0, 1)
#if not is_adversarial(perturbed):
    #scaled_a *= -1
us = 0  #unsuccess

for i in range(100):
    #sr = ep.normal(x_advs, x_advs.shape)
    sr = ep.normal(x_advs, (1, 3, x_advs.shape[2] // 4, x_advs.shape[3] // 4))
    #print(sr.shape)
    sr = ep.astensor(F.interpolate(sr.raw, scale_factor=4, mode='bilinear', align_corners=False))
    sr /= ep.norms.l2(atleast_kd(flatten(sr), sr.ndim)) + 1e-12
    sr = atleast_kd(delta, sr.ndim) * sr
    # print("shape")
    # print(sr.shape)
    perturbed = x_advs + sr
    perturbed = ep.clip(perturbed, 0, 1)
    sr = perturbed - x_advs
    if not is_adversarial(perturbed):
        sr *= -1
    vals.append(sr)

grad = ep.mean(ep.stack(vals, 0), 0)
grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
alpha = 1.'''
'''print(self.advd)
if self.advd is not None and self.last is not None:
    if self.advd < 0.2:
        grad = alpha * grad + (1-alpha) * self.last
        #print(1)'''

'''grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
grad = atleast_kd(delta, grad.ndim) * grad
grad = grad + scaled_a'''
'''k = 2.
while True:
    perturbed = x_advs + grad + k * scaled_a
    perturbed = ep.clip(perturbed, 0, 1)
    if not is_adversarial(perturbed):
        print("-------------------------------")
        print(is_adversarial(x_advs))
        # print(torch.cosine_similarity(grad.raw.view(-1), scaled_a.raw.view(-1), 0))
        k /= 5
        grad /= 2
        # grad = phi * grad - (1-phi) * scaled_a#lowa
    else:
        #scaled_a = perturbed - x_advs - grad
        grad = perturbed - x_advs
        break'''
# grad = grad + scaled_a
# lowa = F.interpolate(scaled_a.raw, scale_factor=0.25, mode='area')
# lowa = ep.astensor(F.interpolate(lowa, scale_factor=4, mode='bilinear', align_corners=False))

# grad = -scaled_a
# grad = vals[0]
'''grad = None
maxx = -10000
for val in vals:
    cl = torch.cosine_similarity(scaled_a.raw.view(-1), val.raw.view(-1), 0).item()
    if cl > maxx:
        grad = val
        maxx = cl'''
# grad = abs(maxx)*grad - scaled_a
# print("maxx: ", end='')
# print(maxx)
# print(grad.shape)
# grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12