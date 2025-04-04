import logging
from typing import Union, Any, Optional, Callable, List
from typing_extensions import Literal

import math

import eagerpy as ep
import numpy as np
import torch
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.tensorboard import TensorBoard
from ..models import Model

from ..criteria import Criterion

from ..distances import l1

from ..devutils import atleast_kd, flatten

from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from ..distances import l2, linf
import cv2
import random


class Block(MinimizationAttack):
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
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.initial_num_evals = initial_gradient_eval_steps
        self.max_num_evals = max_gradient_eval_steps
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.tensorboard = tensorboard
        self.constraint = constraint

        assert constraint in ("l2", "linf")
        if constraint == "l2":
            self.distance = l2
        else:
            self.distance = linf

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

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                #init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                #将一个block初始化为随机噪声，其它部位不变。
                init_attack = BlockInitAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)

            x_advs, rect_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            x_advs = ep.astensor(starting_points)

        is_adv = is_adversarial(x_advs)
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
        x_advs = self._binary_search(is_adversarial, originals, x_advs)

        assert ep.all(is_adversarial(x_advs))

        distances = self.distance(originals, x_advs)

        for step in range(self.steps):
            print("step ", end="")
            print(step)
            delta = self.select_delta(originals, distances, step)

            # Choose number of gradient estimation steps.
            num_gradient_estimation_steps = int(
                min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals])
            )
            gradients = self.approximate_gradients_rect(
                is_adversarial, x_advs, num_gradient_estimation_steps, delta, rect_advs
            )

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
                    success = is_adversarial(x_advs_proposals)
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)

                    if ep.all(success):
                        break

                # Update the sample.
                x_advs = ep.clip(
                    x_advs + atleast_kd(epsilons, update.ndim) * update, 0, 1
                )

                assert ep.all(is_adversarial(x_advs))

                # Binary search to return to the boundary.
                x_advs = self._binary_search(is_adversarial, originals, x_advs)

                assert ep.all(is_adversarial(x_advs))

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

                    mask = is_adversarial(x_advs_proposals)

                    x_advs_proposals = self._binary_search(
                        is_adversarial, originals, x_advs_proposals
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
            tb.histogram("norms", distances, step)

        return restore_type(x_advs)

    def approximate_gradients(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        x_advs: ep.Tensor,
        steps: int,
        delta: ep.Tensor,
        rect_advs
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))
        ''' print("noise_shape")
        print(noise_shape)
        while True:
            pass'''
        if self.constraint == "l2":
            rv = ep.normal(x_advs, noise_shape)
        elif self.constraint == "linf":
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12

        scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

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

    def approximate_gradients_rect(
            self,
            is_adversarial: Callable[[ep.Tensor], ep.Tensor],
            x_advs: ep.Tensor,
            steps: int,
            delta: ep.Tensor,
            rect_advs,
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))
        # rv = ep.zeros(x_advs, noise_shape)
        rv = np.zeros(noise_shape)
        '''if self.constraint == "l2":
            rv = ep.normal(x_advs, noise_shape)
        elif self.constraint == "linf":
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)'''
        N = x_advs.shape[0]
        assert x_advs.shape[1] == 3
        for i in range(N):
            x, y, w, h = rect_advs[i]
            rv_rect = np.random.normal(size=(steps, 3, w, h))
            rv[:, i, :, x:x+w, y:y+h] = rv_rect

        rv = torch.from_numpy(rv).to(torch.float32)
        if torch.cuda.is_available():
            rv = rv.cuda()
        rv = ep.astensor(rv)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12

        scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = is_adversarial(perturbed[step])
            multipliers_list.append(
                ep.where(
                    decision,
                    ep.ones(x_advs, (len(x_advs, ))),
                    -ep.ones(x_advs, (len(decision, ))),
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

        return grad


from .base import FlexibleDistanceMinimizationAttack
from ..distances import Distance
import warnings


class BlockInitAttack(FlexibleDistanceMinimizationAttack):
    """Blends the input with a normal noise input until it is misclassified.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        directions : Number of random directions in which the perturbation is searched.
        steps : Number of blending steps between the original image and the random
            directions.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        directions: int = 1000,
        steps: int = 1000,
    ):
        super().__init__(distance=distance)
        self.directions = directions
        self.steps = steps

        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        X, restore_type = ep.astensor_(inputs)

        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        is_adversarial = get_is_adversarial(criterion_, model)

        min_, max_ = model.bounds

        N = len(X)

        cv2.setUseOptimized(True)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        if torch.cuda.is_available():
            pass
            #X = restore_type(X).cpu()
        if X.shape[1] == 3:
            X = np.moveaxis(X.numpy(), 1, 3)
        else:
            X = X.numpy()
        import matplotlib.pyplot as plt
        #选择k个框
        k = 200
        rect_x = np.zeros(shape=(k, X.shape[0]), dtype=np.int)
        rect_y = np.zeros(shape=(k, X.shape[0]), dtype=np.int)
        rect_w = X.shape[2] * np.ones(shape=(k, X.shape[0]), dtype=np.int)
        rect_h = X.shape[2] * np.ones(shape=(k, X.shape[0]), dtype=np.int)
        img_id = -1
        for img in X:
            img_id += 1
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()
            rects = np.array(sorted(rects, key=lambda x:(x[2]*x[3])))
            if rects.shape[0] > k:
                stp = rects.shape[0] // k
                rects = rects[::stp]
            for i, rect in (enumerate(rects)):
                if i == k-1:
                    break
                x, y, w, h = rect
                img_cpy = img.copy()
                img_cpy[x:x+w, y:y+h] = 0.
                #init_candidate = ep.astensor(torch.from_numpy(np.expand_dims(np.expand_dims(np.moveaxis(img_cpy, 2, 0), 0), 0)))
                #init_candidate = ep.astensor(torch.from_numpy(np.moveaxis(img_cpy, 2, 0)))
                init_candidate = np.moveaxis(img_cpy, 2, 0)
                #print(init_candidate.shape)
                rect_x[i, img_id] = x
                rect_y[i, img_id] = y
                rect_w[i, img_id] = w
                rect_h[i, img_id] = h

        is_adv = np.array([False] * X.shape[0])
        init_advs = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))    #(2, 3, 224, 224)
        rect_advs = np.zeros(shape=(X.shape[0], 4), dtype=np.int)
        #noise_advs = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
        for i in range(k):
            xs, ys, ws, hs = rect_x[i], rect_y[i], rect_w[i], rect_h[i]
            init_candidates = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
            #noise_candidates = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
            for j in range(X.shape[0]):
                img = X[j]
                #print("img:")
                #print(img.shape)
                x, y, w, h = xs[j], ys[j], ws[j], hs[j]
                img_cpy = img.copy()
                random_ = np.random.normal(random.uniform(-1, 2), random.uniform(0, 1), (w, h, 3))
                random_ = np.where(random_ >= min_, random_, min_)
                random_ = np.where(random_ <= max_, random_, max_)
                #img_cpy[x:x + w, y:y + h, :] = 0.

                img_cpy[x:x + w, y:y + h, :] = random_
                #noise_candidates[j, x:x + w, y:y + h, :] = random_
                init_candidate = np.moveaxis(img_cpy, 2, 0)
                init_candidates[j, :, :, :] = init_candidate

            candi_tensor = torch.from_numpy(init_candidates).to(torch.float32)
            if torch.cuda.is_available():
                candi_tensor = candi_tensor.cuda()
            candi_tensor = ep.astensor(candi_tensor)
            # print(candi_tensor.shape)
            adv_xo = is_adversarial(candi_tensor)
            if torch.cuda.is_available():
                #adv_xo = adv_xo.cpu()
                pass
            if adv_xo.any():
                change_idx = adv_xo.numpy() & (np.logical_not(is_adv))
                is_adv = is_adv | adv_xo.numpy()
                init_advs[change_idx, :, :, :] = init_candidates[change_idx]
                #noise_advs[change_idx, :, :, :] = noise_candidates[change_idx]
                #print(rect_advs[change_idx, 0])
                rect_advs[change_idx, 0] = rect_x[i, change_idx]
                rect_advs[change_idx, 1] = rect_y[i, change_idx]
                rect_advs[change_idx, 2] = rect_w[i, change_idx]
                rect_advs[change_idx, 3] = rect_h[i, change_idx]
                #print(adv_xo.numpy())

                '''plot_advs = init_candidates[adv_xo.numpy()]
                for plot_adv in plot_advs:
                    plt.imshow(np.moveaxis(plot_adv, 0, 2))
                    plt.show()'''

                if is_adv.all():
                    break

        #plot images
        '''for img in init_advs:
            print(img.shape)
            plt.imshow(np.moveaxis(img, 0, 2))
            plt.show()'''

        print("oh yes!")
        if not is_adv.all():
            warnings.warn("Failed to initialize adversarial images.")
        print("===========")
        ret = torch.from_numpy(init_advs).to(torch.float32)

        if torch.cuda.is_available():
            ret = ret.cuda()
        return ep.astensor(ret), rect_advs


from skimage import transform
class SmoothAttack(FlexibleDistanceMinimizationAttack):
    """Blends the input with a normal noise input until it is misclassified.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        directions : Number of random directions in which the perturbation is searched.
        steps : Number of blending steps between the original image and the random
            directions.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        directions: int = 1000,
        steps: int = 1000,
    ):
        super().__init__(distance=distance)
        self.directions = directions
        self.steps = steps

        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        X, restore_type = ep.astensor_(inputs)

        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        is_adversarial = get_is_adversarial(criterion_, model)

        min_, max_ = model.bounds

        N = len(X)

        cv2.setUseOptimized(True)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        if torch.cuda.is_available():
            pass
            #X = restore_type(X).cpu()
        if X.shape[1] == 3:
            X = np.moveaxis(X.numpy(), 1, 3)
        else:
            X = X.numpy()
        import matplotlib.pyplot as plt
        #选择k个框
        k = 200
        rect_x = np.zeros(shape=(k, X.shape[0]), dtype=np.int)
        rect_y = np.zeros(shape=(k, X.shape[0]), dtype=np.int)
        rect_w = X.shape[2] * np.ones(shape=(k, X.shape[0]), dtype=np.int)
        rect_h = X.shape[2] * np.ones(shape=(k, X.shape[0]), dtype=np.int)
        img_id = -1
        for img in X:
            img_id += 1
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()
            rects = np.array(sorted(rects, key=lambda x:(x[2]*x[3])))
            if rects.shape[0] > k:
                stp = rects.shape[0] // k
                rects = rects[::stp]
            for i, rect in (enumerate(rects)):
                if i == k-1:
                    break
                x, y, w, h = rect
                img_cpy = img.copy()
                img_cpy[x:x+w, y:y+h] = 0.
                #init_candidate = ep.astensor(torch.from_numpy(np.expand_dims(np.expand_dims(np.moveaxis(img_cpy, 2, 0), 0), 0)))
                #init_candidate = ep.astensor(torch.from_numpy(np.moveaxis(img_cpy, 2, 0)))
                init_candidate = np.moveaxis(img_cpy, 2, 0)
                #print(init_candidate.shape)
                rect_x[i, img_id] = x
                rect_y[i, img_id] = y
                rect_w[i, img_id] = w
                rect_h[i, img_id] = h

        is_adv = np.array([False] * X.shape[0])
        init_advs = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))    #(2, 3, 224, 224)
        rect_advs = np.zeros(shape=(X.shape[0], 4), dtype=np.int)
        #noise_advs = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
        for i in range(k):
            xs, ys, ws, hs = rect_x[i], rect_y[i], rect_w[i], rect_h[i]
            init_candidates = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
            #noise_candidates = np.zeros(shape=(X.shape[0], X.shape[3], X.shape[2], X.shape[1]))
            for j in range(X.shape[0]):
                img = X[j]
                #print("img:")
                #print(img.shape)
                x, y, w, h = xs[j], ys[j], ws[j], hs[j]
                img_cpy = img.copy()
                random_ = np.random.normal(random.uniform(-1, 2), random.uniform(0, 1), (w, h, 3))
                random_ = np.where(random_ >= min_, random_, min_)
                random_ = np.where(random_ <= max_, random_, max_)
                shape = random_.transpose(1, 2, 0).shape
                factor = 4
                random_ = transform.resize(random_.transpose(1, 2, 0), (shape[0]//factor, shape[1], shape[2]//factor)).transpose(2, 0, 1)
                random_ = transform.resize(random_.transpose(1, 2, 0), shape).transpose(2, 0, 1)
                #img_cpy[x:x + w, y:y + h, :] = 0.

                img_cpy[x:x + w, y:y + h, :] = random_
                #noise_candidates[j, x:x + w, y:y + h, :] = random_
                init_candidate = np.moveaxis(img_cpy, 2, 0)
                init_candidates[j, :, :, :] = init_candidate

            candi_tensor = torch.from_numpy(init_candidates).to(torch.float32)
            if torch.cuda.is_available():
                candi_tensor = candi_tensor.cuda()
            candi_tensor = ep.astensor(candi_tensor)
            # print(candi_tensor.shape)
            adv_xo = is_adversarial(candi_tensor)
            if torch.cuda.is_available():
                #adv_xo = adv_xo.cpu()
                pass
            if adv_xo.any():
                change_idx = adv_xo.numpy() & (np.logical_not(is_adv))
                is_adv = is_adv | adv_xo.numpy()
                init_advs[change_idx, :, :, :] = init_candidates[change_idx]
                #noise_advs[change_idx, :, :, :] = noise_candidates[change_idx]
                #print(rect_advs[change_idx, 0])
                rect_advs[change_idx, 0] = rect_x[i, change_idx]
                rect_advs[change_idx, 1] = rect_y[i, change_idx]
                rect_advs[change_idx, 2] = rect_w[i, change_idx]
                rect_advs[change_idx, 3] = rect_h[i, change_idx]
                #print(adv_xo.numpy())

                '''plot_advs = init_candidates[adv_xo.numpy()]
                for plot_adv in plot_advs:
                    plt.imshow(np.moveaxis(plot_adv, 0, 2))
                    plt.show()'''

                if is_adv.all():
                    break

        #plot images
        '''for img in init_advs:
            print(img.shape)
            plt.imshow(np.moveaxis(img, 0, 2))
            plt.show()'''

        print("oh yes!")
        if not is_adv.all():
            warnings.warn("Failed to initialize adversarial images.")
        print("===========")
        ret = torch.from_numpy(init_advs).to(torch.float32)

        if torch.cuda.is_available():
            ret = ret.cuda()
        return ep.astensor(ret), rect_advs