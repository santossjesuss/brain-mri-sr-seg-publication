from abc import ABC
from pathlib import Path
from sklearn.model_selection import GroupKFold
from experiments.experiment import Experiment
from enums.resolution_enum import Resolution
from pipelines.super_resolution_pipeline import SuperResolutionPipeline
from pipelines.segmentation_pipeline import SegmentationPipeline
from pipelines.frozen_sr_frozen_seg_pipeline import FrozenSRFrozenSegPipeline
from pipelines.frozen_sr_trainable_seg_pipeline import FrozenSRTrainableSegPipeline
from pipelines.trainable_sr_frozen_seg_pipeline import TrainableSRFrozenSegPipeline
from pipelines.joint_sr_seg_e2e_pipeline import JointSRSegE2EPipeline
from pipelines.joint_sr_seg_combined_pipeline import JointSRSegCombinedPipeline
from pipelines.sequential_joint_sr_first_pipeline import SequentialJointSRFirstPipeline
from pipelines.sequential_joint_seg_first_pipeline import SequentialJointSegFirstPipeline

class BaseExperiments(ABC):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config

        lesion_training_dataset = dataset(
            is_training=True, 
            is_control_dataset=False,
            view=self.config.view, 
            scale_factor=self.config.scale_factor,
            target_width=self.config.target_width,
            target_height=self.config.target_height
        )
        self.full_train_dataset = lesion_training_dataset

        self.folds = self._generate_patient_folds(
            names=self.full_train_dataset.image_names,
            num_folds=self.config.num_folds
        )

        self.test_dataset = dataset(
            is_training=False,
            is_control_dataset=False,
            view=self.config.view, 
            scale_factor=self.config.scale_factor,
            target_width=self.config.target_width,
            target_height=self.config.target_height
        )

    def _generate_patient_folds(self, names, num_folds):
        # indices = list(range(len(names)))
        patient_id_groups = [
            self._get_patient_id(name)
            for name in names
        ]

        splitter = GroupKFold(n_splits=num_folds)
        folds = list(splitter.split(
            # X=indices,
            X=names,
            groups=patient_id_groups
        ))

        return folds

    def _get_patient_id(self, filename):
        filename_parts = self._parse_mri_filename(filename)
        patient_id = filename_parts[0]

        return patient_id

    def _parse_mri_filename(filename):
        parts = Path(filename).stem.split("_")

        if len(parts != 3):
            raise ValueError(f"Unexpected filename {filename}")

        patient_id = parts[1]
        slice_id = parts[2]

        return patient_id, slice_id

    def _create_experiment(self, name, pipeline, data_resolution=None):
        return Experiment(
            config=self.config,
            name=name,
            pipeline=pipeline,
            folds=self.folds,
            full_train_dataset=self.full_train_dataset,
            test_dataset=self.test_dataset,
            dataset_name=self.config.dataset_name,
            data_resolution=data_resolution
        )

    def get_super_resolution(self):
        return self._create_experiment(
            name=self.config.sr_name, 
            pipeline=SuperResolutionPipeline
        )

    def get_hr_segmentation(self):
        return self._create_experiment(
            name=self.config.hr_seg_name, 
            pipeline=SegmentationPipeline,
            data_resolution=Resolution.HR
        )

    def get_lr_segmentation(self):
        return self._create_experiment(
            name=self.config.lr_seg_name, 
            pipeline=SegmentationPipeline,
            data_resolution=Resolution.LR
        )

    def get_frozen_sr_frozen_seg(self):
        return self._create_experiment(
            name=self.config.frozen_sr_frozen_seg, 
            pipeline=FrozenSRFrozenSegPipeline
        )

    def get_frozen_sr_trainable_seg(self):
        return self._create_experiment(
            name=self.config.frozen_sr_trainable_seg, 
            pipeline=FrozenSRTrainableSegPipeline
        )

    def get_trainable_sr_frozen_seg(self):
        return self._create_experiment(
            name=self.config.trainable_sr_frozen_seg, 
            pipeline=TrainableSRFrozenSegPipeline
        )

    # This pipeline uses Segmentation Loss to learn
    def get_joint_sr_seg_e2e(self):
        return self._create_experiment(
            name=self.config.joint_sr_seg_e2e, 
            pipeline=JointSRSegE2EPipeline
        )

    # This pipeline uses both SuperRes and Segmentation Losses to learn
    def get_joint_sr_seg_combined(self):
        return self._create_experiment(
            name=self.config.joint_sr_seg_combined, 
            pipeline=JointSRSegCombinedPipeline
        )

    def get_sequential_joint_sr_first(self):
        return self._create_experiment(
            name=self.config.sequential_joint_sr_first,
            pipeline=SequentialJointSRFirstPipeline
        )

    def get_sequential_joint_seg_first(self):
        return self._create_experiment(
            name=self.config.sequential_joint_seg_first,
            pipeline=SequentialJointSegFirstPipeline
        )