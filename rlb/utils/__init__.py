import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import hydra
import os.path as osp
import sys
from omegaconf import DictConfig, OmegaConf

def configurable(config_path="config/default.yaml"):
    def wrapper(main_func):
        config_path_arg = None
        i = 1
        for a in sys.argv[1:]:
            if a.startswith("config="):
                config_path_arg = a.split("=")[-1]
                sys.argv.pop(i)
                i += 1
                break
        if config_path_arg is None:
            config_path_arg = config_path
        assert config_path_arg, "config file must be given by `config=path/to/file`"
        main_wrapper = hydra.main(config_path=osp.abspath(osp.dirname(config_path_arg)),
                                config_name=osp.splitext(osp.basename(config_path_arg))[0],
                                version_base=None)
        return main_wrapper(main_func)
    return wrapper



def load_hydra_config(config_path, overrides=[]):

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path=osp.dirname(config_path), job_name="load_config"):
        cfg = compose(config_name=osp.splitext(osp.basename(config_path))[0], overrides=overrides)
    return cfg


def config_to_dict(cfg):
    return OmegaConf.to_container(cfg)

