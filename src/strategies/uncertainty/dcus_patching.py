import torch
import time
import warnings

import numpy as np
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.engine.model import Model
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss
from ultralytics.utils import LOGGER, colorstr, RANK, TQDM, dist, __version__ # type: ignore
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, autocast, unset_deterministic
from ultralytics.nn.tasks import load_checkpoint, ClassificationModel, DetectionModel, SegmentationModel, PoseModel, OBBModel

def newV8__call__(self, preds, batch):
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
    # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
    # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

    loss[0] *= self.hyp.box  # box gain
    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain
    
    classwise_quality = []
    with torch.no_grad():
        # fg_mask: (B, N_anchors), target_bboxes: (B, N_anchors, 4), target_scores: (B, N_anchors, nc)
        # pred_scores: (B, N_anchors, nc), pred_bboxes: (B, N_anchors, 4)
        for b in range(batch_size):
            fg = fg_mask[b]
            if fg.sum() == 0:
                continue
            # For each positive anchor, get assigned class, pred score, pred box, target box
            assigned_classes = target_scores[b][fg].argmax(1)  # [num_pos]
            pred_conf = pred_scores[b][fg, assigned_classes].sigmoid()  # [num_pos]
            pred_box = pred_bboxes[b][fg]  # [num_pos, 4]
            tgt_box = target_bboxes[b][fg]  # [num_pos, 4]
            ious = bbox_iou(pred_box, tgt_box, xywh=False, CIoU=False).diag()  # [num_pos]
            quality = torch.pow(pred_conf, self.hyp.get("dcus_xi", 0.6)) * torch.pow(ious, 1. - self.hyp.get("dcus_xi", 0.6))
            classwise_quality.append(torch.stack([assigned_classes.float(), quality], dim=1))
    if classwise_quality:
        classwise_quality = torch.cat(classwise_quality, dim=0)
    else:
        classwise_quality = torch.zeros((0, 2), device=self.device)
    return loss * batch_size, loss.detach(), classwise_quality

def new_init_criterion(self):
    return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

def state_dict(self, *args, **kwargs):
    # Get the state dict from the underlying model
    state = self.model.state_dict(*args, **kwargs)
    # Add classwise_quality if it exists
    model_to_check = self.model.module if hasattr(self.model, "module") else self.model
    if hasattr(model_to_check, "classwise_quality"):
        state["classwise_quality"] = model_to_check.classwise_quality.cpu()
    return state

def load_state_dict(self, state_dict, strict=True):
    # Restore classwise_quality if present
    if "classwise_quality" in state_dict:
        model_to_set = self.model.module if hasattr(self.model, "module") else self.model
        model_to_set.classwise_quality = state_dict["classwise_quality"]
        # Remove from state_dict to avoid PyTorch warnings
        state_dict = {k: v for k, v in state_dict.items() if k != "classwise_quality"}
    self.model.load_state_dict(state_dict, strict=strict)

def update_classwise_quality_ema(self, class_quality_sum, class_quality_count, base_momentum=0.999):
    if not hasattr(self, "class_quality"):
        self.class_quality = torch.zeros_like(class_quality_sum)
    avg_qualities = class_quality_sum / (class_quality_count + 1e-6)
    self.class_quality = base_momentum * self.class_quality + (1.0 - base_momentum) * avg_qualities

def _do_train(self, world_size=1):
    ensure_dcus_patching()
    
    if world_size > 1:
        self._setup_ddp(world_size)
        ensure_dcus_patching()
    self._setup_train()


    nb = len(self.train_loader)
    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
        f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    self.optimizer.zero_grad()

    num_classes = self.model.nc if hasattr(self.model, "nc") else 80

    self.class_quality = torch.zeros(num_classes, device=self.device)
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scheduler.step()

        self._model_train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        if epoch == (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        if RANK in {-1, 0}:
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None

        class_quality_sum = torch.zeros(num_classes, device=self.device)
        class_quality_count = torch.zeros(num_classes, device=self.device)

        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

            # Forward
            with autocast(self.amp):
                batch = self.preprocess_batch(batch)
                loss, self.loss_items, classwise_quality = self.model(batch)
                self.loss = loss.sum()
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )

            if classwise_quality.numel() > 0:
                classes = classwise_quality[:, 0].long()
                qualities = classwise_quality[:, 1]
                for c in range(num_classes):
                    mask = (classes == c)
                    class_quality_sum[c] += qualities[mask].sum()
                    class_quality_count[c] += mask.sum()

            # Backward
            self.scaler.scale(self.loss).backward()

            # Optimize
            if ni - last_opt_step >= self.accumulate:
                self.optimizer_step()
                last_opt_step = ni

                # Timed stopping
                if self.args.time:
                    self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    if RANK != -1:
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0) # type: ignore
                        self.stop = broadcast_list[0]
                    if self.stop:
                        break

            # Log
            if RANK in {-1, 0}:
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                pbar.set_description( # type: ignore
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                    )
                )
                self.run_callbacks("on_batch_end")
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")

        self.update_classwise_quality_ema(class_quality_sum, class_quality_count, base_momentum=0.999)
        
        model_to_set = self.model.module if hasattr(self.model, "module") else self.model
        setattr(model_to_set, "classwise_quality", self.class_quality.detach().clone())
        if hasattr(self.model, "module"):
            setattr(self.model, "classwise_quality", self.class_quality.detach().clone())
        
        if RANK in {-1, 0}:
            try:
                save_dir = getattr(self, 'save_dir', None)
                classwise_quality_path = Path(save_dir) / "classwise_quality.npy" # type: ignore
                np.save(classwise_quality_path, self.class_quality.detach().cpu().numpy())
            except Exception as e:
                print(f'DCUS: Failed to save epoch-level classwise_quality: {e}')

        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
        self.run_callbacks("on_train_epoch_end")
        if RANK in {-1, 0}:
            final_epoch = epoch + 1 >= self.epochs
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "classwise_quality"])

            # Validation
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            # Save model
            if self.args.save or final_epoch:
                self.save_model()
                self.run_callbacks("on_model_save")

        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        if self.args.time:
            mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
            self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            self._setup_scheduler()
            self.scheduler.last_epoch = self.epoch
            self.stop |= epoch >= self.epochs
        self.run_callbacks("on_fit_epoch_end")
        if self._get_memory(fraction=True) > 0.5:
            self._clear_memory()

        if RANK != -1:
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0) # type: ignore
            self.stop = broadcast_list[0]
        if self.stop:
            break
        epoch += 1

    if RANK in {-1, 0}:
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks("on_train_end")
    self._clear_memory()
    unset_deterministic()
    self.run_callbacks("teardown")

def save_model(self):
    import io
    
    ensure_dcus_patching()
    
    buffer = io.BytesIO()
    model_to_save = self.model.module if hasattr(self.model, "module") else self.model
    
    classwise_quality = getattr(model_to_save, "classwise_quality", None)
    
    if classwise_quality is None:
        classwise_quality = getattr(self.model, "classwise_quality", None)
        
    if classwise_quality is None and hasattr(self.ema.ema, "classwise_quality"):
        classwise_quality = getattr(self.ema.ema, "classwise_quality", None)
    
    torch_save_dict = {
        "epoch": self.epoch,
        "best_fitness": self.best_fitness,
        "model": None,
        "ema": deepcopy(self.ema.ema).half(),
        "updates": self.ema.updates,
        "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
        "train_args": vars(self.args),
        "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
        "train_results": self.read_results_csv(),
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch_save_dict["classwise_quality"] = classwise_quality.cpu() if classwise_quality is not None else None
    torch.save(torch_save_dict, buffer)
    serialized_ckpt = buffer.getvalue()
    self.last.write_bytes(serialized_ckpt)
    
    if classwise_quality is not None:
        classwise_quality_path = self.last.parent / "classwise_quality.npy"
        np.save(classwise_quality_path, classwise_quality.cpu().numpy())
    else:
        print(f'DCUS: classwise_quality is None, not saving numpy file')
    
    if self.best_fitness == self.fitness:
        self.best.write_bytes(serialized_ckpt)
        if classwise_quality is not None:
            best_classwise_quality_path = self.best.parent / "classwise_quality.npy"
            np.save(best_classwise_quality_path, classwise_quality.cpu().numpy())
    if (self.save_period > 0) and (self.epoch % self.save_period == 0):
        epoch_path = self.wdir / f"epoch{self.epoch}.pt"
        epoch_path.write_bytes(serialized_ckpt)
        if classwise_quality is not None:
            epoch_classwise_quality_path = epoch_path.parent / f"epoch{self.epoch}_classwise_quality.npy"
            np.save(epoch_classwise_quality_path, classwise_quality.cpu().numpy())
    
    print("Classwise quality:", classwise_quality)

def setup_model(self):
    if isinstance(self.model, torch.nn.Module):
        return

    cfg, weights = self.model, None
    ckpt = None
    if str(self.model).endswith(".pt"):
        weights, ckpt = load_checkpoint(self.model)
        cfg = weights.yaml
    elif isinstance(self.args.pretrained, (str, Path)):
        weights, _ = load_checkpoint(self.args.pretrained)
    self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)

    if ckpt is not None and "classwise_quality" in ckpt:
        model_to_set = self.model.module if hasattr(self.model, "module") else self.model
        setattr(model_to_set, "classwise_quality", ckpt["classwise_quality"].clone())

    return ckpt

def ensure_dcus_patching():
    v8DetectionLoss.__call__ = newV8__call__ # type: ignore
    yolo.detect.DetectionTrainer._do_train = _do_train # type: ignore
    yolo.detect.DetectionTrainer.setup_model = setup_model # type: ignore
    yolo.detect.DetectionTrainer.save_model = save_model # type: ignore
    yolo.detect.DetectionTrainer.update_classwise_quality_ema = update_classwise_quality_ema # type: ignore
    Model.state_dict = state_dict # type: ignore
    Model.load_state_dict = load_state_dict # type: ignore

def patching(model):
    v8DetectionLoss.__call__ = newV8__call__ # type: ignore
    Model.state_dict = state_dict
    Model.load_state_dict = load_state_dict # type: ignore
    print(f"Patching model at RANK {RANK}")
    ensure_dcus_patching()
    yolo_instance = model.model
    class NewYOLO(YOLO):
        def __init__(self, *args, **kwargs):
            # Ensure patching is applied every time we create a YOLO instance
            print(f"NewYOLO.__init__ called at RANK {RANK}")
            ensure_dcus_patching()
            super().__init__(*args, **kwargs)
        
        @property
        def task_map(self) -> dict[str, dict[str, Any]]:
            # Ensure patching before returning task map
            ensure_dcus_patching()
            return {
                "classify": {
                    "model": ClassificationModel,
                    "trainer": yolo.classify.ClassificationTrainer,
                    "validator": yolo.classify.ClassificationValidator,
                    "predictor": yolo.classify.ClassificationPredictor,
                },
                "detect": {
                    "model": DetectionModel,
                    "trainer": yolo.detect.DetectionTrainer,
                    "validator": yolo.detect.DetectionValidator,
                    "predictor": yolo.detect.DetectionPredictor,
                },
                "segment": {
                    "model": SegmentationModel,
                    "trainer": yolo.segment.SegmentationTrainer,
                    "validator": yolo.segment.SegmentationValidator,
                    "predictor": yolo.segment.SegmentationPredictor,
                },
                "pose": {
                    "model": PoseModel,
                    "trainer": yolo.pose.PoseTrainer,
                    "validator": yolo.pose.PoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                },
                "obb": {
                    "model": OBBModel,
                    "trainer": yolo.obb.OBBTrainer,
                    "validator": yolo.obb.OBBValidator,
                    "predictor": yolo.obb.OBBPredictor,
                },
            }

    yolo_instance.__class__ = NewYOLO
    ensure_dcus_patching()
    print("Done patching model.")
    return model

if __name__ == "__main__":
    model = YOLO('yolo11n.pt')
    model = patching(model)
    print(model.model.task_map["detect"]["trainer"].update_classwise_quality_ema) # type: ignore
