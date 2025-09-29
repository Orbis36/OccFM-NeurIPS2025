from forecast.models.model_template import ModelTemplate
from .occfm_vae import OccFmVAE
from .occfm_vae3d import OccFmVAE3D

__all__ = {
    'ModelTemplate' : ModelTemplate,
    'OccFmVAE' : OccFmVAE,
    'OccFmVAE3D': OccFmVAE3D
}


def build_compressor(model_cfg, loss_cfg):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, loss_cfg=loss_cfg
    )
    return model