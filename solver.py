import logging
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.config import instantiate
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    AMPTrainer,
    default_setup,
    default_writers,
    hooks)


def do_test(cfg, model):
    # Test function called by the EvalHook when current_iter = eval_period
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(cfg, a):
    
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
    trainer.register_hooks(
        [   
            hooks.IterationTimer(), #Track the time spent for each iteration and get a summary at the end
            hooks.LRScheduler(optimizer=optim, scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) #A checkpointer that can save/load model as well as extra checkpointable objects.
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter( #Writes events on log_period 
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True) #Resume=True to load pre-trained or checkpoint file
    if checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)