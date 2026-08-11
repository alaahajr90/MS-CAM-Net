"""Training and evaluation loops for MS-CAM-Net Stage 3."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from metrics_stage3 import classification_metrics


def _collect_metrics(state_true, state_pred, state_prob, level_true, level_pred, level_prob):
    return {
        "state": classification_metrics(state_true, state_pred, state_prob),
        "level": classification_metrics(level_true, level_pred, level_prob),
    }


def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps: int = 4):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = 0.0
    state_true, state_pred, state_prob = [], [], []
    level_true, level_pred, level_prob = [], [], []

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        clip = batch["clip"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        level = batch["level"].to(device, non_blocking=True)
        rppg = batch["rppg"].to(device, non_blocking=True)

        outputs = model(clip)
        total, _ = criterion(outputs, state, level, rppg)
        (total / accumulation_steps).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(total.detach().cpu())
        state_p = torch.softmax(outputs["state"].detach(), dim=1)
        level_p = torch.softmax(outputs["level"].detach(), dim=1)
        state_true.extend(state.cpu().numpy())
        state_pred.extend(state_p.argmax(dim=1).cpu().numpy())
        state_prob.extend(state_p.cpu().numpy())
        level_true.extend(level.cpu().numpy())
        level_pred.extend(level_p.argmax(dim=1).cpu().numpy())
        level_prob.extend(level_p.cpu().numpy())

    return {
        "loss": running / max(len(loader), 1),
        "metrics": _collect_metrics(
            np.asarray(state_true), np.asarray(state_pred), np.asarray(state_prob),
            np.asarray(level_true), np.asarray(level_pred), np.asarray(level_prob),
        ),
    }


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    state_true, state_pred, state_prob = [], [], []
    level_true, level_pred, level_prob = [], [], []

    for batch in tqdm(loader, desc="Validation", leave=False):
        clip = batch["clip"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        level = batch["level"].to(device, non_blocking=True)
        rppg = batch["rppg"].to(device, non_blocking=True)

        outputs = model(clip)
        total, _ = criterion(outputs, state, level, rppg)
        running += float(total.cpu())

        state_p = torch.softmax(outputs["state"], dim=1)
        level_p = torch.softmax(outputs["level"], dim=1)
        state_true.extend(state.cpu().numpy())
        state_pred.extend(state_p.argmax(dim=1).cpu().numpy())
        state_prob.extend(state_p.cpu().numpy())
        level_true.extend(level.cpu().numpy())
        level_pred.extend(level_p.argmax(dim=1).cpu().numpy())
        level_prob.extend(level_p.cpu().numpy())

    return {
        "loss": running / max(len(loader), 1),
        "metrics": _collect_metrics(
            np.asarray(state_true), np.asarray(state_pred), np.asarray(state_prob),
            np.asarray(level_true), np.asarray(level_pred), np.asarray(level_prob),
        ),
    }
