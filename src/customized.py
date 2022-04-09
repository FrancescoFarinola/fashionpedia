from detectron2.engine.hooks import HookBase
import logging
import datetime
import time
from collections import abc
from contextlib import ExitStack
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.evaluation.evaluator import inference_context, DatasetEvaluator, DatasetEvaluators

class EarlyStop(HookBase):
    def __init__(self, checkpointer, eval_period, patience, delta, val_metric):
        self._logger = logging.getLogger(__name__)
        self._checkpointer = checkpointer
        self._period = eval_period
        self._val_metric = val_metric
        self._patience = patience
        self._delta = delta
    
    def _checking(self):
        metrics = self.trainer.storage.history(self._val_metric).values()
        print(metrics)
        if metrics is None:
            self._logger.warning(f"Given val metric {self._val_metric} does not seem to be computed/stored.")
            return
        else: 
            # metrics are in format[(4.374246668072073, 99), (5.19200121426791, 199), (6.227405385224193, 300) 
            #we unzip the tuples and get only the first values of tuples
            metric_values = list(zip(*metrics))[0]
            patience_losses = metric_values[-self._patience:] #keep the last _patience losses
            ref_loss = metric_values[-self._patience-1] #keep the loss value previous to the patience losses to compare
            if all(self._delta > (k - ref_loss) for k in patience_losses):
                self._logger.info(f"Saved last model before terminating: {patience_losses[-1]} @ {self.trainer.iter} steps")
                self._checkpointer.save("model_final")
                self._logger.info(f"Best model saved @ {list(zip(*metrics))[1][-self._patience-1]} steps with {self._val_metric} : {ref_loss} ")
                raise SystemExit("Model has not improved! Terminating...") # a bit violent but let us terminate the training
        

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
            and next_iter > self._period * self._patience #start using this hook after epoch 'patience' + 1 
        ):
            self._checking()   
        
    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._checking()



def inference(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    This is the same inference_on_datasets function from detectron2.evaluation.evaluator
    but with less logs and less stats to be calculated to make it less noisier with 
    printing results.
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time
            start_data_time = time.perf_counter()

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results