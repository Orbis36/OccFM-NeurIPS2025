from collections import defaultdict

def accumulate_disp_dict(val_disp_dicts):
    total = defaultdict(float)
    count = len(val_disp_dicts)

    for d in val_disp_dicts:
        for k, v in d.items():
            total[k] += v.item()  # tensor -> float

    avg_dict = {k: total[k] / count for k in total}
    return avg_dict