import argparse
import datetime
import os
import signal
import socket
import sys
import time
import warnings
from typing import Any

import torch
import torch.distributed as dist
import yaml
from matplotlib import pyplot as plt
from rangerlite import RangerLite
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_, get_total_norm

from allshowers import data_loader, data_sets, flow_matching, transformer, util
from . import data_util
from tqdm.auto import tqdm

import wandb

from dotenv import load_dotenv

class Trainer:
    def __init__(
        self,
        conf: dict[str, Any],
        device: str | torch.device = "cpu",
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        use_wandb: bool = False,
    ) -> None:
        self.conf = conf
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        self.num_epochs = conf["train"]["num_epochs"]
        self.learning_rate = conf["train"]["learning_rate"]
        self.batch_size = conf["train"]["batch_size"]
        self.weight_decay = conf["train"].get("weight_decay", 0)
        self.momentum = conf["train"].get("momentum", 0)
        self.optimizer_name = conf["train"].get(
            "optimizer", "AdamW" if self.weight_decay > 0 else "Adam"
        )
        self.scheduler_name = conf["train"].get("scheduler", None)
        self.grad_clip = conf["train"].get("grad_clip", None)
        self.grad_accum = conf["train"].get("grad_accum", 1)
        self.result_path = conf["result_path"]
        self.batch_size = (self.batch_size + self.world_size - 1) // self.world_size

        self.checkpoint_file = self.get_path("checkpoints/last.pt")
        self.best_file = self.get_path("weights/best.pt")
        self.final_file = self.get_path("weights/final.pt")
        self.plot_folder = "plots/"

        self.use_wandb = use_wandb
        
        # trafos_file = self.get_path("preprocessing/trafos.pt")
        trafos_file = "trafos/trafos.pt" ## Hardcoding the trafo for now, not like we change training data everyday :)

        self.init_model(conf["model"])
        if "num_layers" in conf["model"]:
            conf["data"]["num_layers"] = conf["model"]["num_layers"]

        print("Buildind the train and val loaders")
######################################################################################################
######################################################################################################
######################################################################################################
        # self.train_loader, self.val_loader, self.trafos = data_sets.get_data_loaders(
        #     conf["data"],
        #     self.batch_size,
        #     self.rank,
        #     self.world_size,
        #     self.local_rank,
        #     trafos_file,
        # )
        # print("Loaded the data")

        start = conf["data"]["start"] if "start" in conf["data"] else 0
        stop = conf["data"]["stop"] if "stop" in conf["data"] else 0

        total_showers = stop - start

        if "val_len" in conf["data"]:
            val_len = conf["data"]["val_len"]
        else:
            if rank == 0:
                warnings.warn("val_len not found in config setting to 20 % of training")
                val_len = int(0.2*total_showers)
                
        showers_per_rank = total_showers // self.world_size

        (energy_scaler,
         coordinate_scaler,
         conditioning_scaler) = data_util.ShowerDatasetFromH5.PrepareScalesFromTrafoFile(config_dataset = conf["data"],
                                                                                         trafo_path=trafos_file,
                                                                                         rank=self.rank,
                                                                                         world_size=self.world_size,
                                                                                         local_rank=self.local_rank
                                                                                        )
        
        self.train_dataset = data_util.ShowerDatasetFromH5(path = conf["data"]["path"],
                                                          energy_scaler=energy_scaler,
                                                          coordinate_scaler=coordinate_scaler,
                                                          conditioning_scaler = conditioning_scaler,
                                                          start = start + rank * showers_per_rank,
                                                          stop = start + (rank+1) * showers_per_rank,
                                                          num_layers=conf["model"]["num_layers"])
        
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=conf["train"]["batch_size"]//world_size,
                                           shuffle=True,
                                           num_workers=16,
                                           persistent_workers=True,
                                           prefetch_factor = 4
                                          )

        if rank == 0:

            self.test_dataset = data_util.ShowerDatasetFromH5(path = conf["data"]["path"],
                                                              energy_scaler=energy_scaler,
                                                              coordinate_scaler=coordinate_scaler,
                                                              conditioning_scaler = conditioning_scaler,
                                                              start = stop,
                                                              stop = stop + val_len,
                                                              num_layers=conf["model"]["num_layers"])

            self.test_dataloader = DataLoader(self.test_dataset,
                                              batch_size=conf["train"]["batch_size"]//self.world_size,
                                              shuffle=True,
                                              num_workers=4,
                                              persistent_workers=True,
                                              prefetch_factor = 4
                                          )
        if world_size > 1:
            torch.distributed.barrier(device_ids=[self.local_rank])
        # self.trafos = {key: value.to(self.device) for key, value in self.trafos.items()}
######################################################################################################
######################################################################################################

        self.configure_optimizer()

        if self.rank == 0:
            number_of_parameters = sum(
                p.numel() for p in self.flow.parameters() if p.requires_grad
            )
            print(f"number of parameters: {number_of_parameters}")
            print("train samples:", len(self.train_dataset))
            print("val samples:", len(self.test_dataset))
            print("batch size:", self.batch_size)
            print("gradient accumulation:", self.grad_accum)
            print("num epochs:", self.num_epochs)
            print()
            print(self.flow)
            sys.stdout.flush()

        self.train_losses = []
        self.val_losses = []
        self.train_losses_batch = []
        self.learning_rates = []
        self.grad_norms = []
        self.scores = []
        self.epoch = 0
        self.step = 0
        self.killed = False
        self.min_val_loss = float("inf")
        self.min_score = float("inf")

        if os.path.exists(self.checkpoint_file):
            self.load()

    def init_model(self, model_config: dict[str, Any]) -> None:
        if "flow_config" in model_config:
            flow_config = model_config.pop("flow_config")
        else:
            flow_config = {}
        network = transformer.Transformer(**model_config)
        if self.device.type == "cuda":
            network = torch.compile(network)
        flow = flow_matching.CNF(network, **flow_config)  # type: ignore
        flow = flow.to(self.device)
        if self.world_size > 1:
            flow.network = DDP(flow.network, device_ids=[self.device.index])  # type: ignore
        self.flow = flow

    def configure_optimizer(self) -> None:
        optimizer_name = self.optimizer_name.lower().strip()
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                params=self.flow.network.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                params=self.flow.network.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                params=self.flow.network.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "ranger":
            self.optimizer = RangerLite(
                params=self.flow.network.parameters(),
                lr=self.learning_rate,
                betas=(0.95, 0.999),
                eps=1e-5,
                weight_decay=self.weight_decay,
                lookahead_steps=6,
                lookahead_alpha=0.5,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_name} not implemented."
            )

        if self.scheduler_name is None:
            self.scheduler = None
            self.scheduler_interval = "never"
        elif self.scheduler_name.lower() == "Step".lower():
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=self.num_epochs // 3, gamma=0.1
            )
            self.scheduler_interval = "epoch"
        elif self.scheduler_name.lower() == "Exponential".lower():
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=3e-3 ** (1.0 / self.num_epochs)
            )
            self.scheduler_interval = "epoch"
        elif self.scheduler_name.lower() == "OneCycle".lower():
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.num_epochs
                * (len(self.train_loader) // self.grad_accum),
            )
            self.scheduler_interval = "step"
        elif self.scheduler_name.lower() == "Cosine".lower():
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=self.num_epochs
            )
            self.scheduler_interval = "epoch"
        elif self.scheduler_name.lower() == "CosineWarmup".lower():
            warmup_epochs = 1
            self.scheduler = optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        optimizer=self.optimizer,
                        start_factor=0.1,
                        total_iters=warmup_epochs
                        * (len(self.train_loader) // self.grad_accum),
                    ),
                    optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=self.optimizer,
                        T_max=(self.num_epochs - warmup_epochs)
                        * (len(self.train_loader) // self.grad_accum),
                    ),
                ],
                milestones=[
                    warmup_epochs * (len(self.train_loader) // self.grad_accum)
                ],
            )
            self.scheduler_interval = "step"
        else:
            raise NotImplementedError(
                f"Scheduler {self.scheduler_name} not implemented."
            )

    def init_path(self) -> None:
        if "result_path" not in self.conf or not os.path.isdir(
            self.conf["result_path"]
        ):
            raise ValueError("result_path not found in config.")
        self.result_path = self.conf["result_path"]

    def get_path(self, relative_path: str) -> str:
        path = os.path.join(self.result_path, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_loss(self, batch: data_loader.ModelInputDict) -> torch.Tensor:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        losses = self.flow.loss(**batch)
        losses = losses * batch["mask"].to(losses.dtype)
        losses = torch.mean(losses, dim=(1, 2))
        return losses

    def fit(self) -> None:
        for epoch in range(self.epoch + 1, self.num_epochs + 1):
            self.epoch_start = time.perf_counter()
            self.epoch = epoch
            train_loss_sum = 0.0
            train_loss_count = 0
            self.flow.train()
            self.optimizer.zero_grad()
            for step, batch in tqdm(enumerate(self.train_dataloader),total = len(self.train_dataloader)):
                if step // self.grad_accum == len(self.train_dataloader) // self.grad_accum:
                    continue
                self.step = step
                losses = self.get_loss(batch)
                loss = torch.mean(losses)
                loss.backward()
                self.grad_norms.append(
                    get_total_norm(
                        p.grad for p in self.flow.parameters() if p.grad is not None
                    )
                )
                if (step + 1) % self.grad_accum == 0:
                    if self.grad_clip:
                        clip_grad_norm_(self.flow.parameters(), self.grad_clip)
                    self.optimizer.step()
                    if self.scheduler_interval == "step" and self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                train_loss_sum += loss.item() * len(losses)
                train_loss_count += len(losses)
                
                self.train_losses_batch.append(loss.item())
                self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            if self.scheduler_interval == "epoch" and self.scheduler is not None:
                self.scheduler.step()
            if self.rank == 0:
                self.train_losses.append(train_loss_sum / train_loss_count)
                self.evaluate_and_save()

    def evaluate_and_save(self) -> None:
        self.evaluate()
        try:
            self.print_and_plot()
        except OSError as e:
            print("Plotting failed.")
            print(repr(e))
            sys.stderr.flush()
        try:
            self.save()
        except OSError as e:
            print("Saving failed.")
            print(repr(e))
            sys.stderr.flush()

    @torch.no_grad()
    def evaluate(self) -> None:
        self.flow.eval()
        loss_sum = 0.0
        num_samples = 0
        for batch in self.test_dataloader:
            losses = self.get_loss(batch)
            loss = torch.mean(losses)
            loss_sum += loss.item() * len(losses)
            num_samples += len(losses)
        self.val_losses.append(loss_sum / num_samples)

    def print_and_plot(self) -> None:
        print()
        print(f"======  epoch {self.epoch:4d}  ======")
        print(f"{'time:':16s} {time.perf_counter() - self.epoch_start:.2f}s")
        print(f"{'train loss:':16s} {self.train_losses[-1]:.4f}")
        print(f"{'validation loss:':16s} {self.val_losses[-1]:.4f}")
        print(f"{'learning rate:':16s} {self.learning_rates[-1]:.2e}")
        
        if self.use_wandb:
            wandb.log(
                {
                    "Epoch":self.epoch,
                    "Time taken per epoch":time.perf_counter() - self.epoch_start,
                    "Training Loss":self.train_losses[-1],
                    "Validation Loss":self.val_losses[-1],
                    "Learning Rate":self.learning_rates[-1],
                }
            )
        sys.stdout.flush()

        plt.plot(list(range(self.epoch)), self.train_losses, label="train")
        plt.plot(list(range(self.epoch)), self.val_losses, label="validation")
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(self.get_path(self.plot_folder + "losses.pdf"), bbox_inches="tight")
        plt.close()

        plt.plot(list(range(len(self.learning_rates))), self.learning_rates)
        plt.ylabel("lr")
        plt.xlabel("step")
        plt.savefig(self.get_path(self.plot_folder + "lr.pdf"), bbox_inches="tight")
        plt.close()

    def __signal_handler(self, sig, frame):
        self.killed = True

    def save(self) -> None:
        # ignore interruptions while writing checkpoints
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigtherm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.__signal_handler)
        signal.signal(signal.SIGTERM, self.__signal_handler)

        # save losses
        # overwrite instead of append to avoid incomplete files
        # in case of interruption
        with open(self.get_path("data/losses.txt"), "w") as f:
            if self.scores:
                for i in range(len(self.train_losses)):
                    f.write(
                        f"{self.train_losses[i]} {self.val_losses[i]} {self.scores[i]}\n"
                    )
            else:
                for i in range(len(self.train_losses)):
                    f.write(f"{self.train_losses[i]} {self.val_losses[i]}\n")
        with open(self.get_path("data/losses_batch.txt"), "w") as f:
            for loss, grad_norm, lr in zip(
                self.train_losses_batch, self.grad_norms, self.learning_rates
            ):
                f.write(f"{loss} {grad_norm} {lr}\n")

        # save checkpoint
        if self.scheduler is None:
            scheduler_state_dict = {}
        else:
            scheduler_state_dict = self.scheduler.state_dict()
        flow_state_dict = self.flow.state_dict()
        for key in list(flow_state_dict.keys()):
            if "module." in key:
                flow_state_dict[key.replace("module.", "")] = flow_state_dict.pop(key)
        checkpoint = {
            "flow": flow_state_dict,
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_losses_batch": self.train_losses_batch,
            "scores": self.scores,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
            "min_val_loss": self.min_val_loss,
            "min_score": self.min_score,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": scheduler_state_dict,
        }
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        torch.save(checkpoint, self.checkpoint_file)

        if self.val_losses[-1] < self.min_val_loss:
            self.min_val_loss = self.val_losses[-1]
            if os.path.exists(self.best_file):
                os.remove(self.best_file)
            torch.save(flow_state_dict, self.best_file)

        if bool(self.scores) and (self.scores[-1] < self.min_score):
            self.min_score = self.scores[-1]

        if self.num_epochs == self.epoch:
            torch.save(flow_state_dict, self.final_file)

        # reinstate interruption handlers
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigtherm_handler)
        if self.killed:
            print("exit")
            sys.exit(0)

    def load(self) -> None:
        old_weights = self.flow.state_dict()
        try:
            checkpoint = torch.load(
                self.checkpoint_file, map_location=self.device, weights_only=True
            )
            if isinstance(self.flow.network, DDP):
                for key in list(checkpoint["flow"].keys()):
                    if "network." in key:
                        checkpoint["flow"][
                            key.replace("network.", "network.module.")
                        ] = checkpoint["flow"].pop(key)
            self.flow.load_state_dict(checkpoint["flow"], strict=True)
        except Exception:
            warnings.warn(
                f"Loading {os.path.basename(self.checkpoint_file)} failed. Deleting it."
            )
            sys.stdout.flush()
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            self.flow.load_state_dict(old_weights)
            return

        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_losses_batch = checkpoint["train_losses_batch"]
        self.scores = checkpoint["scores"]
        self.learning_rates = checkpoint["learning_rates"]
        self.grad_norms = checkpoint["grad_norms"]
        self.min_val_loss = checkpoint["min_val_loss"]
        self.min_score = checkpoint["min_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            f"[rank={self.rank}]: Loaded {self.checkpoint_file} at epoch {self.epoch}."
        )
        sys.stdout.flush()


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train network")
    parser.add_argument("param_file",
                        type=str,
                        help="where to find the parameters")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="",
        help="which device to use, if not set, use cuda if available, else cpu",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="where to store results"    
    )
    parser.add_argument(
        "--wandb",
        type=lambda value: value.lower() in ("true", "1", "yes"),
        default=True
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        default=False,
        help="whether or not to use fast development run",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="whether or not to use distributed data parallel",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    with open(parsed_args.param_file) as f:
        conf = yaml.safe_load(f)

    if parsed_args.ddp:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=90))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = rank % torch.cuda.device_count()
        local_rank = device_id
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
    elif parsed_args.device:
        rank = 0
        world_size = 1
        local_rank = 10
        device = parsed_args.device
        if "cuda:" in device:
            torch.cuda.set_device(int(device.split(":")[1]))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    use_wandb = parsed_args.wandb and rank == 0
    if use_wandb:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
             project = "AllShowers",
             # name = f"Default_run"
             )
        
    
    if "result_path" not in conf or parsed_args.fast_dev_run:
        print("result_path not in conf")
        if rank == 0:
            conf["result_path"] = parsed_args.result_path
        if world_size > 1:
            store = dist.TCPStore(  # type: ignore
                host_name=os.environ["MASTER_ADDR"],
                port=12345,
                world_size=world_size,
                is_master=(rank == 0),
            )
            if rank == 0:
                store.set("result_path", conf["result_path"])
            else:
                store.wait(["result_path"])
                conf["result_path"] = store.get("result_path").decode()

    torch.set_float32_matmul_precision("high")

    if rank == 0:
        print("node:", socket.gethostname())
        if device.type == "cuda":
            print("device name:", torch.cuda.get_device_name(device))
        print("device:", str(device))
        print("result_path:", conf["result_path"])
        print("pwd:", os.getcwd())
        print("commit:", os.popen("git rev-parse HEAD").read().strip())
        sys.stdout.flush()

    if parsed_args.fast_dev_run:
        conf["train"]["num_epochs"] = 2
        conf["train"]["batch_size"] = 2
        conf["data"]["val_len"] = 2
        conf["data"]["stop"] = conf["data"]["val_len"] + 4 * world_size
    
    trainer = Trainer(conf, device, rank, world_size, local_rank, use_wandb = use_wandb)
    
    print(
        f"[rank={rank}] Start training on device {device} on host {socket.gethostname()}."
    )
    sys.stdout.flush()
    trainer.fit()

    if rank == 0:
        print()
        print("Training finished.")
        sys.stdout.flush()

    if parsed_args.ddp:
        # destroy_process_group has a shorter timeout than barrier
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    load_dotenv()
    main()
