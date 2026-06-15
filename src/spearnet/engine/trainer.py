"""Training engine (Section 5): AdamW + cosine LR + warmup, mixed precision, grad clip,
checkpointing, and per-epoch validation.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..losses import CompoundLoss
from .evaluator import evaluate


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, DataLoader],
        cfg: Config,
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.run.device if (cfg.run.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.model = model.to(self.device)
        self.loaders = loaders

        cw = self._resolve_class_weights(cfg, loaders)
        self.criterion = CompoundLoss(
            num_classes=cfg.num_classes,
            w_ce=cfg.loss.w_ce,
            w_dice=cfg.loss.w_dice,
            w_tversky=cfg.loss.w_tversky,
            w_boundary=cfg.loss.w_boundary,
            w_edge=cfg.loss.w_edge,
            w_focal=cfg.loss.w_focal,
            focal_gamma=cfg.loss.focal_gamma,
            tversky_alpha=cfg.loss.tversky_alpha,
            tversky_beta=cfg.loss.tversky_beta,
            present_only=cfg.loss.present_only,
            class_weights=cw,
            ignore_index=cfg.loss.ignore_index,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.steps_per_epoch = max(1, len(loaders["train"]))
        self.scheduler = self._build_scheduler()
        self._amp = cfg.optim.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp)

        os.makedirs(cfg.run.out_dir, exist_ok=True)
        self.best_metric = -math.inf
        self.start_epoch = 0
        self.history = []
        if cfg.run.resume:
            self._load_checkpoint(cfg.run.resume)

    def _resolve_class_weights(self, cfg, loaders):
        """Return a per-class weight tensor (manual, auto-estimated, or None)."""
        if cfg.loss.class_weights is not None:
            w = cfg.loss.class_weights
        elif cfg.loss.auto_class_weights and (cfg.loss.w_focal > 0 or cfg.loss.w_ce > 0):
            from ..data.dataset import estimate_class_weights

            train_ds = loaders["train"].dataset
            w = estimate_class_weights(
                train_ds, cfg.num_classes,
                max_samples=cfg.loss.auto_weight_samples, seed=cfg.run.seed,
            )
        else:
            return None
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def _build_scheduler(self):
        total = self.cfg.optim.epochs * self.steps_per_epoch
        warmup = self.cfg.optim.warmup_epochs * self.steps_per_epoch
        min_ratio = self.cfg.optim.min_lr / max(self.cfg.optim.lr, 1e-12)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            prog = (step - warmup) / max(total - warmup, 1)
            cos = 0.5 * (1 + math.cos(math.pi * prog))
            return min_ratio + (1 - min_ratio) * cos

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ---------------------------------------------------------------- training
    def train(self) -> Dict[str, object]:
        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg.optim.epochs):
            self._train_one_epoch(epoch)
            if (epoch + 1) % cfg.run.val_interval == 0 and "val" in self.loaders:
                results = evaluate(self.model, self.loaders["val"], cfg, self.device)
                metric = results.get(cfg.run.save_best_metric, results["miou"])
                self._log_val(epoch, results, metric)
                self._save_checkpoint(epoch, metric, is_best=metric > self.best_metric)
                self.best_metric = max(self.best_metric, metric)
        self._save_checkpoint(cfg.optim.epochs - 1, self.best_metric, is_best=False, name="last.pt")
        return {"best_metric": self.best_metric, "history": self.history}

    def _train_one_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        self.model.train()
        running = 0.0
        t0 = time.time()
        for it, batch in enumerate(self.loaders["train"]):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self._amp):
                outputs = self.model(batch["image"])
                loss, comps = self.criterion(outputs, batch)
                loss = loss / cfg.optim.accumulate

            self.scaler.scale(loss).backward()
            if (it + 1) % cfg.optim.accumulate == 0:
                if cfg.optim.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # Only advance the LR schedule if the optimizer actually stepped
                # (AMP may skip a step while it calibrates the loss scale -> avoids the
                # "lr_scheduler.step() before optimizer.step()" warning).
                if self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()

            running += loss.item() * cfg.optim.accumulate
            if (it + 1) % cfg.run.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                comp_str = " ".join(f"{k}={v.item():.3f}" for k, v in comps.items() if k != "total")
                print(
                    f"[epoch {epoch+1}/{cfg.optim.epochs}] it {it+1}/{self.steps_per_epoch} "
                    f"loss={running/(it+1):.4f} lr={lr:.2e} {comp_str}",
                    flush=True,
                )
        dt = time.time() - t0
        print(f"  epoch {epoch+1} done in {dt:.1f}s avg_loss={running/self.steps_per_epoch:.4f}", flush=True)

    # ---------------------------------------------------------------- logging / ckpt
    def _log_val(self, epoch: int, results: Dict, metric: float) -> None:
        rare = {k: v["iou"] for k, v in results["per_class"].items()}
        print(
            f"  [val] epoch {epoch+1}  mIoU={results['miou']:.4f} "
            f"mF1={results['mean_f1']:.4f} mRecall={results['mean_recall']:.4f} "
            f"pixAcc={results['pixel_acc']:.4f}",
            flush=True,
        )
        print(f"        per-class IoU: " + ", ".join(f"{k}:{v:.3f}" for k, v in rare.items()), flush=True)
        entry = {"epoch": epoch + 1, "metric": metric, **{
            "miou": results["miou"], "mean_f1": results["mean_f1"],
            "mean_recall": results["mean_recall"], "pixel_acc": results["pixel_acc"],
        }}
        self.history.append(entry)
        with open(os.path.join(self.cfg.run.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool, name: Optional[str] = None) -> None:
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg.to_dict(),
        }
        if name:
            torch.save(ckpt, os.path.join(self.cfg.run.out_dir, name))
        if is_best:
            torch.save(ckpt, os.path.join(self.cfg.run.out_dir, "best.pt"))
            print(f"  ✓ new best ({self.cfg.run.save_best_metric}={metric:.4f}) saved", flush=True)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", -math.inf)
        print(f"Resumed from {path} at epoch {self.start_epoch}", flush=True)
