import logging
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.config import instantiate
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    AMPTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch)


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)
    trainer = (AMPTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer,)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(optimizer=optim, scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True)
    if checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)
    checkpointer.save("model_1x")