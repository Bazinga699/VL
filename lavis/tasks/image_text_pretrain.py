"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed as dist
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

@registry.register_task("image_text_pretrain_profiler")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
    
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=1,
            iters_per_epoch= len(data_loader),#(12423374+1388521) // (data_loader._index_sampler.batch_size * dist.get_world_size() ),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        wait = 50
        warmup = 5
        active = 2
        repeat =1
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        tb_handler = torch.profiler.tensorboard_trace_handler('/data/profiler/BLIP2-bg-fb-2')
        
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        with torch.profiler.profile(
            schedule=schedule, on_trace_ready=tb_handler,
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
                # if using iter-based runner, we stop after iters_per_epoch iterations.
                if i >= (wait + warmup + active) * repeat:
                    break

                samples = next(data_loader)

                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )

                lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)

                # after_train_step()
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                    optimizer.zero_grad()

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                if dist.get_rank() == 0 and i % log_freq == 0:
                    for k, meter in metric_logger.meters.items():
                        self.writer.add_scalar(k, meter.value, epoch * iters_per_epoch + i)
                
                prof.step()

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        
        
        
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

