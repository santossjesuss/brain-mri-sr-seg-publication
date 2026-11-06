import os
import random
import torch
from torch.utils.data import Subset
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from pipelines.base_pipeline import BasePipeline
from enums.hyperparameter_enum import HyperparameterVersion
from utils.model_persistence import load_model_for_inference

class SequentialJointSegFirstPipeline(BasePipeline):
    '''
    Prerequisite: Trained SR model to be plugged into Phase 1
    Structure: Two phases
        First   - Frozen SR -> Trainable Seg    (train SR first)
        Second  - Trainable SR -> Frozen Seg    (then plug SR, freeze it and train Seg)
    Loss: Segmentation Loss across both phases
    
    Note: Fist phase (Trainable SR -> Frozen SR) is already in the system, so SR is extracted and plugged into
    the second phase of the training (Frozen SR -> Trainable Seg) in this class. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, full_train_dataset, folds):
        version = HyperparameterVersion.V1
        fold_validation_results = []

        for fold_idx, (train_indices, validation_indices) in enumerate(folds):
            print(f"\nFold {fold_idx + 1}/{self.config.num_folds}")

            train_dataset = Subset(full_train_dataset, train_indices)
            validation_dataset = Subset(full_train_dataset, validation_indices)

            train_loader = self._get_dataloader(train_dataset)
            validation_loader = self._get_dataloader(validation_dataset)

            sr_model = self._init_rcan()
            seg_model = self._init_unet()
            criterion = self._get_seg_loss(version=version)
            validation_metrics = self._get_seg_validation_metrics()
            logger = self._get_logger()

            # Frozen SR -> Trainable Seg
            first_phase_sr_seg = MultiStageModel(
                sr_model,
                seg_model,
                freeze_stage_1=True,
                freeze_stage_2=True
            )
            first_phase_path = os.path.join(self.config.folder_name, f'{self.config.frozen_sr_trainable_seg}_fold{fold_idx}.pth')
            load_model_for_inference(model=first_phase_sr_seg, saving_name=first_phase_path)
            fine_tuned_seg_model = first_phase_sr_seg.model_stage_2

            # Trainable SR -> Frozen Seg
            sr_model = self._init_rcan()
            second_phase_sr_seg = MultiStageModel(
                sr_model,
                fine_tuned_seg_model,
                freeze_stage_1=False,
                freeze_stage_2=True
            )
            optimizer = self._get_optimizer(second_phase_sr_seg.parameters(), version)
            scheduler = self._get_scheduler(optimizer)

            fold_adapted_path = self._adapt_saving_path(fold_idx)

            trainer = MultiStageTrainer(
                config=self.config,
                model=second_phase_sr_seg,
                device=self.device,
                train_loader=train_loader,
                validation_loader=validation_loader,
                criterion=criterion,
                validation_metrics=validation_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                saving_name=fold_adapted_path
            )

            _, validation_results, training_time, peak_memory = trainer.train(epochs=self.config.epochs)
            fold_validation_results.append({
                "fold": fold_idx + 1,
                "fold_metrics": validation_results,
                "training_time": training_time,
                "peak_memory": peak_memory
            })

        return fold_validation_results

    def test(self, test_dataset):
        fold_test_results = []

        for fold_idx in self.config.num_folds:
            print(f"\nFold {fold_idx + 1}/{self.config.num_folds}")

            test_loader = self._get_dataloader(test_dataset)

            sr_model = self._init_rcan()
            seg_model = self._init_unet()
            validation_metrics = self._get_seg_validation_metrics()

            sequential_joint_seg_first_model = MultiStageModel(
                sr_model, 
                seg_model, 
                freeze_stage_1=True, 
                freeze_stage_2=True
            )
            fold_adapted_path = self._adapt_saving_path(fold_idx)
            load_model_for_inference(model=sequential_joint_seg_first_model, saving_name=fold_adapted_path)

            trainer = MultiStageTrainer(
                config=self.config,
                model=sequential_joint_seg_first_model,
                device=self.device,
                validation_metrics=validation_metrics
            )

            test_metrics = trainer.test(test_loader)
            fold_test_results.append({
                "fold": fold_idx + 1,
                "fold_metrics": test_metrics
            })

        return fold_test_results

    # PENDING TO DELETE 
    def predict(self, input_tensor):
        sr_model = self._init_rcan()
        seg_model = self._init_unet()

        sequential_joint_seg_first_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=sequential_joint_seg_first_model, saving_name=self.saving_path)
        sequential_joint_seg_first_model.to(self.device).eval()

        _, hr_mask, lr_image, _ = input_tensor
        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = sequential_joint_seg_first_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1)
            predicted_mask = output_mask.squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, hr_mask)

            return {
                'input_image': lr_image,
                'target_mask': hr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }

    def predict_random(self, dataset):
        sr_model = self._init_rcan()
        seg_model = self._init_unet()

        sequential_joint_seg_first_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=sequential_joint_seg_first_model, saving_name=self.saving_path)
        sequential_joint_seg_first_model.to(self.device).eval()

        idx = random.randint(0, len(dataset) - 1)
        _, hr_mask, lr_image, _ = dataset[idx]

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = sequential_joint_seg_first_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1)
            predicted_mask = output_mask.squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, hr_mask)

            return {
                'input_image': lr_image,
                'target_mask': hr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }