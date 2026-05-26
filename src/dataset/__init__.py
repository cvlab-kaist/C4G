import time
from dataclasses import fields

from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg, DatasetRE10kCfgWrapper
from .dataset_spring import DatasetSpring, DatasetSpringCfg, DatasetSpringCfgWrapper
from .dataset_kubric import DatasetKubric, DatasetKubricCfg, DatasetKubricCfgWrapper
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "spring": DatasetSpring,
    "kubric": DatasetKubric,
}


DatasetCfgWrapper = (
    DatasetSpringCfgWrapper
    | DatasetKubricCfgWrapper
    | DatasetRE10kCfgWrapper
)

DatasetCfg = (
    DatasetRE10kCfg
    | DatasetSpringCfg
    | DatasetKubricCfg
)


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        print(f"{cfg.name} dataset initializing with view sampler {cfg.view_sampler}...")
        t_start = time.time()
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)
        print(f"{cfg.name} dataset initialized in {time.time() - t_start:.1f}s")

    return datasets
