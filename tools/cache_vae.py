import torch
import pickle
from rich.console import Group
from forecast.utils.eval_utils import DistributedListBuffer

def cache_model(model, data_loader, model_func, progress, cache_mode=False, live=None,
                mu_sigma_cache=False, save_path=None, console=None, is_main_process=None):

    if not (mu_sigma_cache or cache_mode):
        return 0

    if mu_sigma_cache:
        model.module.quantization.mu_sigma_cache = mu_sigma_cache
        assert save_path is not None
    if cache_mode:
        model.module.quantization.latent_cache = cache_mode
        assert save_path is not None

    for data_frac_loader in data_loader:
        split = 'training' if data_frac_loader.dataset.training else 'validation'
        all_data_save = DistributedListBuffer()

        if is_main_process:
            cache_task = progress.add_task(description=f"Cache {split} set samples", total=len(data_frac_loader))
            if live is not None:
                live.update(progress)

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_frac_loader):
                _, tb_dict, val_disp_dict = model_func(model, batch)

                path_gt = val_disp_dict['gt_path']
                trajectory = val_disp_dict['trajectory'].detach().cpu().numpy()
                if cache_mode:

                    x_sampled = val_disp_dict['x_sampled'].detach().cpu().numpy()
                    all_data_save.append({
                        "x_sampled": x_sampled, "gt_path": path_gt, "gt_trajs": trajectory
                    })

                if mu_sigma_cache:
                    mu = val_disp_dict['mu'].detach().cpu().numpy()
                    sigmas = val_disp_dict['sigmas'].detach().cpu().numpy()
                    all_data_save.append({
                        "mu": mu, "sigmas": sigmas, "gt_path": path_gt, "gt_trajs": trajectory
                    })

                if is_main_process:
                    progress.update(cache_task, advance=1)

        merged_data = all_data_save.gather()
        if is_main_process:
            progress.remove_task(cache_task)
            file_name = save_path + '_' + split + '.pkl'
            console.print(
                f"[bold magenta]✔️ {len(merged_data)} samples in {split} set cached at {file_name}[/bold magenta]"
            )
            with open(file_name, "wb") as f:
                pickle.dump(merged_data, f)


