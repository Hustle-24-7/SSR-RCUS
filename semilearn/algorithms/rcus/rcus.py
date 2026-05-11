# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np
import torch

from .rankup_net import RankUp_Net

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook

from semilearn.core.criterions import CELoss, ClsConsistencyLoss


@ALGORITHMS.register("rcus")
class RCUS(AlgorithmBase):
    """
    RankUp algorithm (https://arxiv.org/abs/2410.22124).

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - arc_ulb_loss_ratio (`float`):
            Weight for unsupervised loss in Arc
        - arc_lb_loss_ratio (`float`):
            Weight for Arc loss
        - T (`float`):
            Temperature for pseudo-label sharpening
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - hard_label (`bool`, *optional*, default to `False`):
            If True, targets have [Batch size] shape with int values. If False, the target is vector
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.init(
            arc_ulb_loss_ratio=args.arc_ulb_loss_ratio,
            arc_lb_loss_ratio=args.arc_lb_loss_ratio,
            T=args.T,
            p_cutoff=args.p_cutoff,
            hard_label=args.hard_label,
            alpha=args.alpha,
        )
        self.ce_loss = CELoss()
        self.cls_consistency_loss = ClsConsistencyLoss()
        super().__init__(args, net_builder, tb_log, logger)

    def init(
        self,
        arc_ulb_loss_ratio,
        arc_lb_loss_ratio,
        T,
        p_cutoff,
        hard_label,
        alpha,
    ):
        self.arc_ulb_loss_ratio = arc_ulb_loss_ratio
        self.arc_lb_loss_ratio = arc_lb_loss_ratio
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.alpha = alpha

    def set_hooks(self):
        super().set_hooks()
        # reset PseudoLabelingHook hook
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

    def set_model(self, **kwargs):
        """
        overwrite the initialize model function
        """
        model = super().set_model(**kwargs)
        model = RankUp_Net(model)
        return model

    def set_optimizer(self, **kwargs):
        """
        set optimizer for algorithm
        """
        from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

        optimizer, scheduler = super().set_optimizer(**kwargs)

        if self.args.sch == "multistep":
            scheduler = MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)

        return optimizer, scheduler

    def set_ema_model(self, **kwargs):
        """
        overwrite the initialize ema model function
        """
        ema_model = self.net_builder(
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
            **kwargs
        )
        ema_model = RankUp_Net(ema_model)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        self.idx_ulb = idx_ulb

        # inference and calculate sup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb, use_arc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            logits_arc_x_lb = outs_x_lb["logits_mat"]
            arc_y_lb = outs_x_lb["targets_mat"]
            logits_arc_x_lb_ = outs_x_lb["logits_arc"]

            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w, use_arc=True)
            feats_x_ulb_w = outs_x_ulb_w["feat"]
            logits_arc_x_ulb_w = outs_x_ulb_w["logits_mat"]
            probs_x_ulb_w = self.compute_prob(logits_arc_x_ulb_w.detach())
            self.bn_controller.unfreeze_bn(self.model)

            outs_x_ulb_s = self.model(x_ulb_s, use_arc=True)
            logits_arc_x_ulb_s = outs_x_ulb_s["logits_mat"]
            feats_x_ulb_s = outs_x_ulb_s["feat"]
            logits_arc_x_ulb_s_ = outs_x_ulb_s["logits_arc"]

            # compute mask
            mask = self.call_hook(
                "masking",
                "MaskingHook",
                logits_x_ulb=probs_x_ulb_w,
                softmax_x_ulb=False,
            )

            # generate unlabeled targets using pseudo label hook
            arc_pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "PseudoLabelingHook",
                logits=probs_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
                softmax=False,
            )

            feat_dict = {
                "x_lb": feats_x_lb,
                "x_ulb_w": feats_x_ulb_w,
                "x_ulb_s": feats_x_ulb_s,
            }

            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")
            # unsup_loss = self.reg_loss(outs_x_ulb_s["logits"], outs_x_ulb_w["logits"], reduction="mean")
            # arc_sup_loss = self.ce_loss(logits_arc_x_lb, arc_y_lb, reduction="mean")
            # arc_unsup_loss = self.cls_consistency_loss(
            #     logits_arc_x_ulb_s, arc_pseudo_label, "ce", mask=mask
            # )

            # arc_loss = self.arc_lb_loss_ratio * arc_sup_loss + self.arc_ulb_loss_ratio * arc_unsup_loss

            reg_loss = sup_loss # + self.ulb_loss_ratio  * unsup_loss

            logits_mixup_x_lb1, logits_mixup_x_lb2 = self.arc_manifold_mixup(feats_x_lb, logits_arc_x_lb_)
            arc_mixup_sup_loss = self.ce_loss(
                logits_mixup_x_lb1, arc_y_lb, reduction="mean"
            ) + self.ce_loss(logits_mixup_x_lb2, arc_y_lb, reduction="mean")
            arc_loss = self.arc_lb_loss_ratio * arc_mixup_sup_loss

            logits_mixup_x_ulb_s1, logits_mixup_x_ulb_s2 = self.arc_manifold_mixup(feats_x_ulb_s, logits_arc_x_ulb_s_)
            arc_mixup_unsup_loss = self.cls_consistency_loss(
                logits_mixup_x_ulb_s1, arc_pseudo_label, "ce", mask=mask
            ) + self.cls_consistency_loss(
                logits_mixup_x_ulb_s2, arc_pseudo_label, "ce", mask=mask)
            arc_loss += self.arc_ulb_loss_ratio * arc_mixup_unsup_loss

            total_loss = reg_loss + arc_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=(sup_loss + arc_mixup_sup_loss).item(),
            unsup_loss=arc_mixup_unsup_loss.item(),
            total_loss=total_loss.item(),
        )
        return out_dict, log_dict

    def arc_manifold_mixup(self, feats_x, logits_arc_x):
        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1.0 - l)

        mixup_feats_x = l * feats_x.unsqueeze(dim=0) + (1.0 - l) * feats_x.unsqueeze(dim=1)
        logits_arc_mixup_x = self.model.arc_classifier(mixup_feats_x)

        # logits_mixup_x1 = (logits_arc_x.unsqueeze(dim=1) - logits_arc_mixup_x).flatten(start_dim=0, end_dim=1)
        # logits_mixup_x2 = (logits_arc_mixup_x.transpose(0, 1) - logits_arc_x.unsqueeze(dim=1)).flatten(start_dim=0, end_dim=1)

        logits_mixup_x1 = (logits_arc_x.unsqueeze(dim=1) - logits_arc_mixup_x.transpose(0, 1)).transpose(0, 1).flatten(start_dim=0, end_dim=1)
        # logits_mixup_x1 = (logits_arc_x.unsqueeze(0) - logits_arc_mixup_x).flatten(start_dim=0, end_dim=1)
        logits_mixup_x2 = (logits_arc_mixup_x - logits_arc_x.unsqueeze(dim=1)).flatten(start_dim=0, end_dim=1)

        return logits_mixup_x1, logits_mixup_x2


    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--arc_ulb_loss_ratio", float, 1.0),
            SSL_Argument("--arc_lb_loss_ratio", float, 1.0),
            SSL_Argument("--T", float, 0.5),
            SSL_Argument("--p_cutoff", float, 0.95),
            SSL_Argument("--hard_label", str2bool, True),
        ]
