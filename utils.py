from os.path import join, expanduser
from collections import defaultdict
import numpy as np
import torch
import sys,os


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_epoch{}.pth".format(epoch))
    torch.save({"state_dict": model.state_dict(),
        		"optimizer": optimizer.state_dict(),
        		"epoch": epoch
        		}, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)