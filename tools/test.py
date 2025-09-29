import torch
import numpy as np
import copy
from rich.console import Group
import torch.distributed as dist

from tools.common_utils.display_utils import format_disp_dict
from tools.common_utils.common_utils import accumulate_disp_dict
from forecast.utils.eval_utils import multi_step_MeanIou, DistributedDictMeanCounter


def setup_occ_comparsion(label_name, frame):

    unique_labels = np.asarray([x for x in range(17)])  # 17 stand for empty
    unique_label_str = [label_name[l] for l in unique_labels]
    IoU_counter = multi_step_MeanIou([1], -100, ['occupied'], 'vox', times=frame)
    IoU_counter.reset()
    mIoU_counter = multi_step_MeanIou(unique_labels, -100, unique_label_str, 'sem', times=frame)
    mIoU_counter.reset()
    return IoU_counter, mIoU_counter


def val_model(model, val_loader, model_func, progress, console_live, use_amp=False, eval_iou=False, eval_fps=False,
              is_main_process=None, rank=None, test_cfm=False):

    label_name = val_loader.dataset.label_name
    model.eval()

    if is_main_process:
        val_task = progress.add_task(description="Eval samples", total=len(val_loader))

    sequence_length = val_loader.dataset.sequence_length
    IoU_counter, mIoU_counter = setup_occ_comparsion(label_name, sequence_length)
    metrics_mean_counter = DistributedDictMeanCounter(rank)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                batch['eval_fps'] = eval_fps
                batch['cfm_eval'] = test_cfm
                val_loss, tb_dict, val_disp_dict = model_func(model, batch)

                if eval_fps:
                    tb_dict["time"] = val_disp_dict["time"]

                metrics_mean_counter.update(tb_dict)

                if eval_iou:
                    if 'gt_occ' not in val_disp_dict:
                        all_seq_gtocc_path = batch['paths'][0][sequence_length:]
                        gt_occ = torch.as_tensor(np.stack([np.load(x[0])['semantics'] if isinstance(x, list)
                                                           else np.load(x)['semantics'] for x in all_seq_gtocc_path])).unsqueeze(0)
                    else:
                        gt_occ = val_disp_dict['gt_occ'].detach().cpu()

                    pred_occ = val_disp_dict['pred_occ'].detach().cpu()

                    if val_loader.dataset.sem_mode:
                        mIoU_counter._after_step(pred_occ, gt_occ)

                        # assume empty is the last label
                        pred_occ[pred_occ != len(label_name)] = 1
                        pred_occ[pred_occ == len(label_name)] = 0
                        gt_occ[gt_occ != len(label_name)] = 1
                        gt_occ[gt_occ == len(label_name)] = 0

                    IoU_counter._after_step(pred_occ, gt_occ)

                    # check real time miou/iou
                    #all_miou, cate_miou = mIoU_counter._after_epoch()
                    #all_iou, cate_iou = IoU_counter._after_epoch()

            if is_main_process:
                progress.update(val_task, advance=1)
                console_live.update(Group(progress, format_disp_dict(tb_dict)))

    dist.barrier(device_ids=[rank])
    all_miou, cate_miou = mIoU_counter._after_epoch()
    all_iou, cate_iou = IoU_counter._after_epoch()
    avg_dict = metrics_mean_counter.compute()

    if is_main_process:
        console_live.update(Group(progress))
        progress.remove_task(val_task)

        avg_dict["all_miou"] = all_miou
        avg_dict["all_iou"] = all_iou

        if val_loader.dataset.sem_mode:
            cate_miou = np.mean(cate_miou, axis=0)
            cate_miou = dict(zip(label_name, cate_miou*100))
            avg_dict['cate_miou'] = cate_miou

    else:
        avg_dict = {}
    return avg_dict

if __name__ == "__main__":
    print()