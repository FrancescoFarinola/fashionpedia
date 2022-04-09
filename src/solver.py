import logging
import torch
import gc
from detectron2.config import instantiate
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    AMPTrainer,
    default_setup,
    default_writers,
    hooks)
from customized import EarlyStop, inference

def do_test(cfg, model):
    # Test function called by the EvalHook when current_iter = eval_period
    if "evaluator" in cfg.dataloader:
        ret = inference(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        return ret

def do_train(cfg, a, args):
    # Setup and initialize necessary variables
    default_setup(cfg, args = a) # Setup environment
    model = instantiate(cfg.model) #Instantiate model
    logger = logging.getLogger("detectron2") #Instantiate logger
    model.to(cfg.train.device) #Set model to be run on cuda

    cfg.optimizer.params.model = model #Correctly set the optimizer
    optim = instantiate(cfg.optimizer) #Instantiate the optimizer
    train_loader = instantiate(cfg.dataloader.train) # instantiate the dataloader for training set
    trainer = (AMPTrainer)(model, train_loader, optim) #Use AMPtrainer to optimize with float16 certain operations
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer,) #Initialize checkpointer

    #Define the training routine - https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/hooks.html for more
    routine = [
         hooks.IterationTimer(),
         hooks.LRScheduler(optimizer=optim, scheduler=instantiate(cfg.lr_multiplier)),
         hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
         hooks.BestCheckpointer(eval_period = cfg.train.eval_period, checkpointer = checkpointer, val_metric = 'segm/AP'),
         EarlyStop(checkpointer = checkpointer, eval_period= cfg.train.eval_period, patience=args.patience, delta=args.delta, val_metric='segm/AP'),
         hooks.PeriodicWriter(default_writers(cfg.train.output_dir, cfg.train.max_iter), period=cfg.train.log_period)
         if comm.is_main_process()
         else None,
         ]
    trainer.register_hooks(routine)
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True) #Resume=True will load a checkpoint in the output dir if available

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train(0, cfg.train.max_iter)