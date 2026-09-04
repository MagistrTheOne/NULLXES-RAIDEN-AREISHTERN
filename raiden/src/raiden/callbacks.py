"""Trainer callbacks: SIGTERM checkpoint, eval curve table, RunPod preemption."""

from __future__ import annotations

import csv
import logging
import os
import signal
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from raiden.paths import logs_dir

logger = logging.getLogger("raiden")


class SigtermCheckpointCallback(TrainerCallback):
    """On SIGTERM/SIGINT, request a checkpoint then stop.

    RunPod sends SIGTERM on pod stop. Saving onto the network volume is the
    only way the run is resumable.
    """

    def __init__(self) -> None:
        self._armed = False
        self._stop_requested = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._armed:
            return
        self._armed = True
        self._prev_term = signal.getsignal(signal.SIGTERM)
        self._prev_int = signal.getsignal(signal.SIGINT)

        def _handler(signum, frame):
            logger.warning("signal %s received — requesting checkpoint + stop", signum)
            self._stop_requested = True
            try:
                if callable(self._prev_term) and signum == signal.SIGTERM:
                    pass
            except Exception:
                pass

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        logger.info("SIGTERM/SIGINT checkpoint handler armed")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self._stop_requested:
            control.should_save = True
            control.should_training_stop = True
            logger.warning("stopping after step %s due to signal", state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if self._stop_requested:
            marker = Path(args.output_dir) / "STOPPED_BY_SIGNAL"
            marker.write_text(f"step={state.global_step}\n", encoding="utf-8")


class EvalCurveCallback(TrainerCallback):
    """Append a row to logs/eval_curve.csv after each eval/save."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = csv_path or (logs_dir() / "eval_curve.csv")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "checkpoint",
                        "global_step",
                        "train_loss",
                        "val_loss",
                        "identity_score",
                        "sycophancy_score",
                        "tone_score",
                        "retention_score",
                        "independence_score",
                        "uncertainty_score",
                    ]
                )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control, metrics=None, **kwargs):
        metrics = metrics or {}
        row = {
            "checkpoint": f"checkpoint-{state.global_step}",
            "global_step": state.global_step,
            "train_loss": _last_train_loss(state),
            "val_loss": metrics.get("eval_loss"),
            "identity_score": metrics.get("eval_identity_score"),
            "sycophancy_score": metrics.get("eval_sycophancy_score"),
            "tone_score": metrics.get("eval_tone_score"),
            "retention_score": metrics.get("eval_retention_score"),
            "independence_score": metrics.get("eval_independence_score"),
            "uncertainty_score": metrics.get("eval_uncertainty_score"),
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [
                    row["checkpoint"],
                    row["global_step"],
                    row["train_loss"],
                    row["val_loss"],
                    row["identity_score"],
                    row["sycophancy_score"],
                    row["tone_score"],
                    row["retention_score"],
                    row["independence_score"],
                    row["uncertainty_score"],
                ]
            )
        logger.info("eval curve row: %s", row)


class LightweightRaidenEvalCallback(TrainerCallback):
    """Run a cheap RAIDEN eval after each checkpoint save.

    Full generation eval is expensive; this callback is gated by
    RAIDEN_EVAL_ON_SAVE=1 (default on) and uses a small prompt subset.
    Failures never abort training — they are logged.
    """

    def __init__(self, eval_fn=None) -> None:
        self.eval_fn = eval_fn
        self.enabled = os.environ.get("RAIDEN_EVAL_ON_SAVE", "1") == "1"

    def on_save(self, args, state: TrainerState, control, model=None, tokenizer=None, **kwargs):
        if not self.enabled or self.eval_fn is None:
            return
        try:
            scores = self.eval_fn(model=model, tokenizer=tokenizer, step=state.global_step)
            logger.info("lightweight RAIDEN eval @ step %s: %s", state.global_step, scores)
            curve = logs_dir() / "eval_curve.csv"
            # scores merged into a sidecar json for later join
            from raiden.logging import write_json

            write_json(
                logs_dir() / f"eval_step_{state.global_step}.json",
                {"step": state.global_step, **(scores or {})},
            )
        except Exception:
            logger.exception("lightweight RAIDEN eval failed at step %s", state.global_step)


def _last_train_loss(state: TrainerState) -> Any:
    for rec in reversed(state.log_history or []):
        if "loss" in rec:
            return rec["loss"]
    return None
