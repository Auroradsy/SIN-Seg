import numpy as np
import os
import random
import pickle
import json
import hydra
# from hydra.core.hydra_config import HydraConfig
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
from pathlib import Path

import torch
from torchsummary import summary
import torch.autograd as autograd

import models
from utils import flatten_configdict, save_visual, dice_loss
from datasets.construct_dataset import dateloader_construct
from losses import sin_seg_loss
from metrics import SegmentationMetrics

# Transunet configs
from models.trans_unet_configs import CONFIGS as vit_configs

# wandb setup
def set_up(cfg):
    if cfg.seed == -1:
        cfg.seed = np.random.randint(0, 100)
    
    # set the seed to have reproducible experiments
    set_seed(cfg.seed)

    # init wandb
    if not cfg.train.state:
        os.environ['WANDB_MODE'] = 'dryrun' # if not training, set wandb to dryrun mode
        os.environ['HYDRA_FULL_ERROR'] = '1' # if not training, set hydra to full error mode
    # wandb.init(
    #     project=cfg.wandb.project, 
    #     entity=cfg.wandb.entity, 
    #     config=flatten_configdict(cfg),
    #     save_code=True,
    #     dir=cfg.wandb.dir,
    # )

# set manual seed, to make the experiment reproducible
def set_seed(
        seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True # True if model and input sizes don't change
        torch.backends.cudnn.deterministic = True

# set the dataloaders and model
def set_dataloaders_and_model(
        cfg: OmegaConf):
    
    # Set GPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.device.index)
        print(f'Using GPU {cfg.device.index}...')
    else:
        print('Using CPU...')

    # set the model
    model_name = cfg.model.name
    model_name = getattr(models, model_name)
    if cfg.model.name == "trans_unet":
        trans_config = vit_configs[cfg.model.vit_name]
        model = model_name(
            config = trans_config,
            img_size=cfg.dataset.size,
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
        )
    elif cfg.model.name in ["missformer", "medt"]:
        model = model_name(
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
            img_size=cfg.dataset.size,
        )
    else:
        model = model_name(
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
        )
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset}, model_name = {model_name}, in_channels = {cfg.dataset.in_channels}, out_chanels = {cfg.dataset.out_channels})"
        )
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) ## not sure if needs p.bias
    print(f"Number of parameters: {num_param}, i.e. {num_param/1e6}M")
    image_sample = torch.randn(1, cfg.dataset.in_channels, cfg.dataset.size, cfg.dataset.size)
    # output_sample = model(image_sample)

    transformer_params = 0
    decoder_params = 0
    seg_head_params = 0
    fusion_align_params = 0
    for name, param in model.named_parameters():
        if ('fusion_align' in name):
            fusion_align_params += param.numel()
            print(f"module: {name}, num_param: {param.numel()}")
        if ('decoder' in name):
            decoder_params += param.numel()
            print(f"module: {name}, num_param: {param.numel()}")
        if ('seg_head' in name):
            seg_head_params += param.numel()
            print(f"module: {name}, num_param: {param.numel()}")
        if ('transformer' in name):
            transformer_params += param.numel()
            print(f"module: {name}, num_param: {param.numel()}")

    # wandb.run.summary["num_param"] =  num_param
    if cfg.device.type == "cuda" and torch.cuda.is_available() and len(cfg.device.index) > 1:
        device = torch.device("cuda:{}".format(cfg.device.index[0]))
        model = torch.nn.DataParallel(model, device_ids=cfg.device.index)
        print(f'Lets use GPU index: {cfg.device.index} in {torch.cuda.device_count()} GPUs!!')
        model = model.to(device)
    elif cfg.device.type == "cuda" and torch.cuda.is_available() and len(cfg.device.index) == 1:
        print(f'Lets use GPU: {cfg.device.index}!')
        device = torch.device("cuda:{}".format(cfg.device.index[0]))
        model = model.to(device)
    elif cfg.device.type == "cpu":
        print('Lets use CPU!')
        model = model.to(cfg.device.type)
    else:
        raise ValueError("Unknown device type")
    
    # torchsummary cannot handle multiple GPUs and just could be used on single cuda:0
    summary(model, (cfg.dataset.in_channels, cfg.dataset.size, cfg.dataset.size))

    # set the dataloaders
    dataloaders = dateloader_construct(cfg)

    return model, dataloaders

# training
def trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cfg: OmegaConf):
    
    # set the optimizer
    optimizer = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }[cfg.train.optimizer]
    optimizer = optimizer(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    print(f"Optimizer: {optimizer}")

    # set the loss function
    loss = {
        "mse": torch.nn.MSELoss,
        "cross_entropy": torch.nn.CrossEntropyLoss,
        #sin_seg loss for weighted BCE+Dice loss
        "sin_seg": sin_seg_loss,
    }[cfg.train.loss.type]
    if cfg.train.loss.type == "sin_seg":
        loss = loss(
            weight=cfg.train.loss.weight,
            num_classes=cfg.dataset.out_channels,
        )
    print(f"Loss function: {loss}")

    # set the metrics
    metrics = {
        "sin_seg": SegmentationMetrics,
    }[cfg.train.loss.type]
    if cfg.train.loss.type == "sin_seg":
        score = metrics()

    # set the scheduler
    scheduler = {
        "step_lr": torch.optim.lr_scheduler.StepLR,
        "cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }[cfg.train.scheduler]
    if cfg.train.scheduler == "step_lr":
        scheduler = scheduler(
            optimizer,
            step_size=cfg.train.scheduler_params.step_size,
            gamma=cfg.train.scheduler_params.gamma,
        )
    print(f"Scheduler: {scheduler}")

    # train
    loss_dict = {
        "train": {
            "bce": {},
            "dice": {},
            "total_loss": {},
        },
        "val": {
            "bce": {},
            "dice": {},
            "total_loss": {},
        },
    }
    metrcs_dict = {
        "train": {
            "f1": {},
            "asd": {},
            "hd95": {},
            "iou": {},
        },
        "val": {
            "f1": {},
            "asd": {},
            "hd95": {},
            "iou": {},
        },
    }
    best_val_loss = np.inf
    best_epoch = 0
    counter = 0
    for epoch in range(cfg.train.epochs):
        print(f"Epoch {epoch+1}/{cfg.train.epochs}")
        for phase in ["train", "val"]:
            bce_loss = 0.0
            dice_loss = 0.0
            total_loss = 0.0
            f1 = 0.0
            asd = 0.0
            hd95 = 0.0
            iou = 0.0
            if phase == "train":
                model.train()
            else:
                model.eval()
            pbar = tqdm(dataloader[phase])
            # for inputs, labels in dataloader[phase]:
            for i, (inputs, labels, indx_name) in enumerate(pbar):
                device = torch.device("cuda:{}".format(cfg.device.index[0]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # loss
                    loss_value = loss(outputs, labels)
                    bce_loss += loss_value['bce'].item() * inputs.size(0)
                    dice_loss += loss_value['dice'].item() * inputs.size(0)
                    total_loss += loss_value['total_loss'].item() * inputs.size(0)
                    if phase == "train":
                        loss_value['total_loss'].backward()
                        optimizer.step()
                    # scores
                    score(outputs, labels)
                    f1 += score.f1_score() * inputs.size(0)
                    asd += score.asd() * inputs.size(0)
                    hd95 += score.hausdorff_95() * inputs.size(0)
                    iou += score.iou_score() * inputs.size(0)

                pbar.set_description(f"{phase} {epoch+1}/{cfg.train.epochs}Epochs|iter{i+1}| bce_loss: {loss_value['bce']:.4f} | dice_loss: {loss_value['dice']:.4f} | total_loss: {loss_value['total_loss']:.4f}")
            
            # after each epoch
            loss_dict[phase]['bce'][epoch+1] = bce_loss / len(dataloader[phase].dataset)
            loss_dict[phase]['dice'][epoch+1] = dice_loss / len(dataloader[phase].dataset)
            loss_dict[phase]['total_loss'][epoch+1] = total_loss / len(dataloader[phase].dataset)
            print(f"---{phase} epoch {epoch+1} | bce_loss: {loss_dict[phase]['bce'][epoch+1]:.4f} | dice_loss: {loss_dict[phase]['dice'][epoch+1]:.4f} | total_loss: {loss_dict[phase]['total_loss'][epoch+1]:.4f}")
            metrcs_dict[phase]['f1'][epoch+1] = f1 / len(dataloader[phase].dataset)
            metrcs_dict[phase]['asd'][epoch+1] = asd / len(dataloader[phase].dataset)
            metrcs_dict[phase]['hd95'][epoch+1] = hd95 / len(dataloader[phase].dataset)
            metrcs_dict[phase]['iou'][epoch+1] = iou / len(dataloader[phase].dataset)
            print(f"---{phase} epoch {epoch+1} | f1_Score: {metrcs_dict[phase]['f1'][epoch+1]:.4f} | asd: {metrcs_dict[phase]['asd'][epoch+1]:.4f} | hd95: {metrcs_dict[phase]['hd95'][epoch+1]:.4f} | iou: {metrcs_dict[phase]['iou'][epoch+1]:.4f}")
            
            # save to wandb
            wandb.log({f"{phase}_bce_loss": loss_dict[phase]['bce'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_dice_loss": loss_dict[phase]['dice'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_total_loss": loss_dict[phase]['total_loss'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_f1": metrcs_dict[phase]['f1'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_asd": metrcs_dict[phase]['asd'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_hd95": metrcs_dict[phase]['hd95'][epoch+1]}, step=epoch+1)
            wandb.log({f"{phase}_iou": metrcs_dict[phase]['iou'][epoch+1]}, step=epoch+1)
        # scheduler
        if cfg.train.scheduler == "reduce_lr_on_plateau":
            scheduler.step(loss_dict['val']['total_loss'][epoch+1])
        else:
            scheduler.step()
        # save the model per frequency
        if (epoch+1) % cfg.train.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_check_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch+1}!")
        # record the best epoch
        if loss_dict['val']['total_loss'][epoch+1] < best_val_loss:
            best_val_loss = loss_dict['val']['total_loss'][epoch+1]
            counter = 0
            best_epoch = epoch+1
        else:
            counter += 1
            print(f"Best model not updated for {counter} epochs!")
            if counter == cfg.train.early_stop:
                print(f"Early stopping at epoch {epoch+1}!")
                break
    # save the best model
    torch.save(model.state_dict(), os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_best_{best_epoch}.pth"))
    print(f"Best model saved at epoch {epoch+1}!")

    # test
    if cfg.dataset.name == 'cell':
        loss_dict['test'] = {
            "bce": {},
            "dice": {},
            "total_loss": {},
        }
        metrcs_dict['test'] = {
            "f1": {},
            "asd": {},
            "hd95": {},
            "iou": {},
        }
        # save_visual(cfg, model, best_epoch, dataloader)
        test_bce_loss = 0.0
        test_dice_loss = 0.0
        test_total_loss = 0.0
        test_f1 = 0.0
        test_asd = 0.0
        test_hd95 = 0.0
        test_iou = 0.0
        
        model.load_state_dict(torch.load(os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_best_{best_epoch}.pth")))
        model.eval()
        pbar = tqdm(dataloader['test'])
        for i, (inputs, labels, indx_name) in enumerate(pbar):
            device = torch.device("cuda:{}".format(cfg.device.index[0]))
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                # loss
                loss_value = loss(outputs, labels)
                test_bce_loss += loss_value['bce'].item() * inputs.size(0)
                test_dice_loss += loss_value['dice'].item() * inputs.size(0)
                test_total_loss += loss_value['total_loss'].item() * inputs.size(0)
                # scores
                score(outputs, labels)
                test_f1 += score.f1_score() * inputs.size(0)
                test_asd += score.asd() * inputs.size(0)
                test_hd95 += score.hausdorff_95() * inputs.size(0)
                test_iou += score.iou_score() * inputs.size(0)

            pbar.set_description(f"test iter{i+1}| bce_loss: {loss_value['bce']:.4f} | dice_loss: {loss_value['dice']:.4f} | total_loss: {loss_value['total_loss']:.4f}")
        loss_dict['test']['bce'] = test_bce_loss / len(dataloader['test'].dataset)
        loss_dict['test']['dice'] = test_dice_loss / len(dataloader['test'].dataset)
        loss_dict['test']['total_loss'] = test_total_loss / len(dataloader['test'].dataset)
        metrcs_dict['test']['f1'] = test_f1 / len(dataloader['test'].dataset)
        metrcs_dict['test']['asd'] = test_asd / len(dataloader['test'].dataset)
        metrcs_dict['test']['hd95'] = test_hd95 / len(dataloader['test'].dataset)
        metrcs_dict['test']['iou'] = test_iou / len(dataloader['test'].dataset)
    else:
        loss_dict['test'] = loss_dict['val']
        metrcs_dict['test'] = metrcs_dict['val']

    # save the loss_dict with json after training
    with open(os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_loss.json"), 'w') as f:
        json.dump(loss_dict, f, indent=4)
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}, bce_loss: {loss_dict['val']['bce'][best_epoch]:.4f}, dice_loss: {loss_dict['val']['dice'][best_epoch]:.4f}, total_loss: {loss_dict['val']['total_loss'][best_epoch]:.4f}")

    # save the metrics_dict with json after training
    with open(os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_metrics.json"), 'w') as f:
        json.dump(metrcs_dict, f, indent=4)
    print(f"Best epoch: {best_epoch}, best f1: {metrcs_dict['val']['f1'][best_epoch]:.4f}, asd: {metrcs_dict['val']['asd'][best_epoch]:.4f}, hd95: {metrcs_dict['val']['hd95'][best_epoch]:.4f}, iou: {metrcs_dict['val']['iou'][best_epoch]:.4f}")

    # train to wandb
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_val_bce_loss"] = loss_dict['val']['bce'][best_epoch]
    wandb.run.summary["best_val_dice_loss"] = loss_dict['val']['dice'][best_epoch]
    wandb.run.summary["best_f1"] = metrcs_dict['val']['f1'][best_epoch]
    wandb.run.summary["best_asd"] = metrcs_dict['val']['asd'][best_epoch]
    wandb.run.summary["best_hd95"] = metrcs_dict['val']['hd95'][best_epoch]
    wandb.run.summary["best_iou"] = metrcs_dict['val']['iou'][best_epoch]
    wandb.save(os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_best_{best_epoch}.pth"))
    # test to wandb
    if cfg.dataset.name == 'cell':
        wandb.run.summary["test_bce_loss"] = loss_dict['test']['bce']
        wandb.run.summary["test_dice_loss"] = loss_dict['test']['dice']
        wandb.run.summary["test_total_loss"] = loss_dict['test']['total_loss']
        wandb.run.summary["test_f1"] = metrcs_dict['test']['f1']
        wandb.run.summary["test_asd"] = metrcs_dict['test']['asd']
        wandb.run.summary["test_hd95"] = metrcs_dict['test']['hd95']
        wandb.run.summary["test_iou"] = metrcs_dict['test']['iou']
    wandb.save(os.path.join(cfg.exp_root, "*.json"))

    # save visual
    if cfg.visualize:
        if cfg.dataset.name != 'cell':
            save_visual(cfg, model, best_epoch, dataloader['val'])
        else:
            save_visual(cfg, model, best_epoch, dataloader['val'])
            save_visual(cfg, model, best_epoch, dataloader['test'])
    


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False) # allow getattr and hasattr methods to be used
    
    running_dir = str(hydra.utils.get_original_cwd())
    working_dir = str(Path.cwd())
    print(f"Running dir: {running_dir}")
    print(f"Working_exp_root: {working_dir}")
    cfg.exp_root = working_dir
    # print input arguments
    print(f"---Input arguments \n{OmegaConf.to_yaml(cfg)}")
    
    set_up(cfg)
    torch.autograd.set_detect_anomaly(True)
    # dataloaders and model
    model, dataloaders = set_dataloaders_and_model(cfg)

    trainer(model, dataloaders, cfg)
    
    

if __name__ == "__main__":
    # initialize()
    # cfg = compose(config_name="./cfg/config.yaml")
    main()