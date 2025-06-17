from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.datasets.pipelines import ObjectSample


@HOOKS.register_module(force=True)
class FadeOjectSampleHook(Hook):
    def __init__(self, num_last_epochs=5, skip_type_keys=("ObjectSample")):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) > runner.max_epochs - self.num_last_epochs:
            runner.logger.info("No ObjectSample now!")
            if hasattr(train_loader.dataset, "dataset"):
                transforms = train_loader.dataset.dataset.pipeline.transforms
            else:
                transforms = train_loader.dataset.pipeline.transforms
            for transform in transforms:
                if isinstance(transform, ObjectSample):
                    transforms.remove(transform)
                    break
            if (
                hasattr(train_loader, "persistent_workers")
                and train_loader.persistent_workers is True
            ):
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
