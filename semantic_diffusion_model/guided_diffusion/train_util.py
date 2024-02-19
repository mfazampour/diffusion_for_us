# this script has been modified to incorporate w&b 
# 07/11/2023

import wandb


import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import gc
import torch
import matplotlib.pyplot as plt

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            num_classes,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            drop_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            output_dir, # Output directory for checkpoints
    ):
        self.model = model
        self.diffusion = diffusion
        #breakpoint()
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        if microbatch > 0: 
            print("Microbatch size: ", self.microbatch)
        else:
            print("Batch size: ", self.batch_size)

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval 
        # After log_interval steps are completed it logs the metrics in logger.
        self.save_interval = save_interval
        # Important to understand how many steps are we doing, to know how often to save.
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        # schedule sampler is uniformly sampling from the diffusion 
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.output_dir = output_dir # or get_blob_logdir() # Store the output directory

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # Initialize W&B
        # wandb.init(project="camus_remote", entity="marinadomin")
        # Optionally add a unique name or additional configuration for the run
        wandb.init(project="modified_echo_from_noise", entity="marina-dominguez", name="First_trial_eps=1_training_1000Diffsteps_128x128")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print("loading model from checkpoint: ", resume_checkpoint)
                self.model.load_state_dict(
                    th.load(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        """
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers 
        for param in self.model.output_blocks.parameters():  
            param.requires_grad = True
        """

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr
    """
    def run_loop(self):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory usage before batch: {torch.cuda.memory_allocated()}")

            batch, cond = next(iter(self.data))
            print(f"Batch size: {batch.size()}")

            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory usage after batch: {torch.cuda.memory_allocated()}")           
        except StopIteration:
            raise ValueError("No data available in DataLoader.")

        cond = self.preprocess_input(cond)
        self.run_step(batch, cond)  # Run one step
        logger.dumpkvs()
    """
     
    # The framework transforms the noise from standard Gaussian distribution 
    # to the realistic image through iterative denoising process. 
    # In each denoising step, we use a U-net-based network to predict noise involved 
    # into the noisy images yt under the guidance of the semantic layouts x.
    
    # for each step noising / denoising we have the og image = batch = y_0 and cond = x = semantic layout
    # where is the noisy image 

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
                # this is at the beginning 0 < 1000, so it will run for 1000 steps
        ):
            # print("Running the run_loop")
            logger.log(f"step: {self.step + self.resume_step}")
            # print("While loop condition: ", self.step + self.resume_step)
            batch, cond = next(self.data) # in this case batch is 1 image and cond is the label map
            # so we have 1 image and 1 label map
            # and the loop will run for 1000 steps in this image with the condition of the label map

            # print(f"Batch size: {batch.size()}") # torch.Size([1, 3, 128, 128])
            # print(f"Cond size: {cond.keys()}") # Cond size: dict_keys(['path', 'label_ori', 'label'])
            # print(f"Conditional label map size: {cond['label'].size()}") # Label size: torch.Size([1, 1, 128, 128])
            # print(f"Og label size: {cond['label_ori'].size()}") Og label size: torch.Size([1, 128, 128])
            # print(f"Path: {cond['path']}")  Path: ['augmented_camus/2CH_ED_augmented/images/training/patient0142_2CH_ED_training_4.png']
            
            cond = self.preprocess_input(cond)
            # print(f"Cond: {cond.keys()}") Cond: dict_keys(['y'])
            # print(f"Condition, label map : {cond['y'].size()}") # Cond: torch.Size([1, 5, 128, 128])
            
            """
            # print the image that we are training on (batch = [1,3,128,128])
            fig, axs = plt.subplots(2, max(batch.shape[1], cond['y'].shape[1]), figsize=(15, 6))

            # Display the image channels
            for i in range(batch.shape[1]):
                axs[0, i].imshow(batch[0,i,:,:].cpu().detach().numpy(), cmap='gray')
                axs[0, i].set_title("Original Image: channel " + str(i))

            # Display the label channels
            for i in range(cond['y'].shape[1]):
                axs[1, i].imshow(cond['y'][0,i,:,:].cpu().detach().numpy(), cmap='gray')
                axs[1, i].set_title("Condition Label Map: channel " + str(i))

            # Remove axes for empty subplots
            for i in range(batch.shape[1], max(batch.shape[1], cond['y'].shape[1])):
                fig.delaxes(axs[0, i])
            for i in range(cond['y'].shape[1], max(batch.shape[1], cond['y'].shape[1])):
                fig.delaxes(axs[1, i])

            plt.tight_layout()
            # wandb.log({"Image with Label": plt})
            """
            
            self.run_step(batch, cond) # Here we are running one step!! 
            # a step in the sense that we are running the forward and backward pass
            # and we are optimizing the loss function
            # so we are training the UNet, updating the parameters and saving the model 

            print("Step: ", self.step + self.resume_step, ". Logging metrics to wandb...")
            
            kvs = logger.getkvs()

            wandb.log(kvs) #, step = self.step + self.resume_step)

            if self.step % self.log_interval == 0:
                print(self.step, " steps completed, let's check the logger")
                logger.dumpkvs() 
            
            if self.step % self.save_interval == 0 and self.step > 0:
                print("Saving step")
                logger.log("Saving step")
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
   
    def run_step(self, batch, cond):
        # print("Running the step")
        logger.log("Running the step: forward, backward, optimize")
        self.forward_backward(batch, cond)
        # loss = self.forward_backward(batch, cond) 
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        # self.log_step(loss)  # pass the loss to log_step

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            """
            estas son las mismas imagenes que tenemos del run_loop

            # print("micro: ", micro.shape) micro:  torch.Size([1, 3, 128, 128])
            # print("micro_cond: ", micro_cond.keys()) micro_cond:  dict_keys(['y'])
            # print("last_batch: ", last_batch) last_batch:  True

            fig, axs = plt.subplots(2, max(micro.shape[1], micro_cond['y'].shape[1]), figsize=(15, 6))

            # Display the image channels
            for i in range(micro.shape[1]):
                axs[0, i].imshow(micro[0,i,:,:].cpu().detach().numpy(), cmap='gray')
                axs[0, i].set_title("Image channel " + str(i))

            # Display the label channels
            for i in range(micro_cond['y'].shape[1]):
                axs[1, i].imshow(micro_cond['y'][0,i,:,:].cpu().detach().numpy(), cmap='gray')
                axs[1, i].set_title("Label channel " + str(i))

            # Remove axes for empty subplots
            for i in range(micro.shape[1], max(micro.shape[1], micro_cond['y'].shape[1])):
                fig.delaxes(axs[0, i])
            for i in range(micro_cond['y'].shape[1], max(micro.shape[1], micro_cond['y'].shape[1])):
                fig.delaxes(axs[1, i])

            plt.tight_layout()
            # wandb.log({"Micro Image and Label Channels": plt})
            """

            # we are sampling from the diffusion process
            # we are sampling the timesteps and the weights
            print("Sampling from the diffusion process: timestep and weights")
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # print(" Current timestep (?) t: ", t) Current timestep (?) t:  tensor([363], device='cuda:0')
            # print("weights: ", weights) weights:  tensor([1.], device='cuda:0')
            
            print("Computing losses: mse, vb and weighted sum of both")

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            # functools.partial in Python is a higher-order function that allows you 
            # to fix a certain number of arguments of a function and generate a new function.

            if last_batch or not self.use_ddp: # last batch is true (and use_ddp is true also)
                losses = compute_losses()
                # print("compute_losses: ", compute_losses)
                """
                losses:  functools.partial(<bound method SpacedDiffusion.training_losses of 
                <guided_diffusion.respace.SpacedDiffusion object at 0x7f296c0571f0>>, 
                DistributedDataParallel(
                    (module): UNetModel(
                        (time_embed): Sequential(
                        (0): Linear(in_features=128, out_features=512, bias=True)
                        (1): SiLU()
                """
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            #breakpoint()

            # {'vb': tensor([0.0182], device='cuda:0', grad_fn=<WhereBackward0>), 
            # 'mse': tensor([0.9790], device='cuda:0', grad_fn=<MeanBackward1>), 
            # 'loss': tensor([0.9790], device='cuda:0', grad_fn=<AddBackward0>)}

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            # and here we have basically trained the first UNet
            # and we have the loss function
            
            # return loss.item()  # return the loss value

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self): #, loss):
        logger.logkv("step", self.step + self.resume_step)
        # wandb.log({"step = step + resume_step": self.step + self.resume_step})
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # wandb.log({"samples = (step + resume_step + 1)*batch": (self.step + self.resume_step + 1) * self.global_batch})
        logger.logkv("lr", self.opt.param_groups[0]['lr'])
        # wandb.log({"lr": self.opt.param_groups[0]['lr']})
        logger.logkv("lr_anneal_steps", self.lr_anneal_steps)
        # wandb.log({"lr_anneal_steps": self.lr_anneal_steps})
        logger.logkv("memory_usage", torch.cuda.memory_allocated())
        # wandb.log({"memory_usage": torch.cuda.memory_allocated()})
        #logger.logkv("loss", loss)
    """
    def log_metrics_to_wandb():
        # Retrieve key-value pairs from the logger
        kvs = logger.getkvs()
        # Log to W&B
        # wandb.log(kvs)
        """
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                save_path = os.path.join(self.output_dir, filename)
                th.save(state_dict, save_path)
                logger.log(f"saved model {rate} to {save_path}")
                # saved 
                """ 
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                    """

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            optimizer_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            optimizer_path = os.path.join(self.output_dir, optimizer_filename)
            th.save(self.opt.state_dict(), optimizer_path)
            """
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
            """
        dist.barrier()

        # Save model checkpoint to W&B
        if dist.get_rank() == 0:
            wandb.save(os.path.join(self.output_dir, f"model{(self.step + self.resume_step):06d}.pt"))
            logger.log("Saved model checkpoint to W&B, model: ", f"model{(self.step + self.resume_step):06d}.pt")


    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = th.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if 'instance' in data:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

        if self.drop_rate > 0.0:
            mask = (th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_semantics = input_semantics * mask

        cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics

        return cond

    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir(self):
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return self.output_dir #logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step :06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # print("Mean of the losses for key: ", key, " is: ", values.mean().item())
        # wandb.log({key: values.mean().item()})
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            #print("Quartile for the current timestep sub_t: ", sub_t, " is: ", quartile)
            #print(f"{key}_q{quartile}", sub_loss)  
            #print("The total number of timesteps is: ", diffusion.num_timesteps)
            #wandb.log({f"{key}_q{quartile}": sub_loss})

            
