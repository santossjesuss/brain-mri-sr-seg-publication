import os
import random
import torch
from torch.utils.data import Subset
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from enums.hyperparameter_enum import HyperparameterVersion
from utils.model_persistence import load_model_for_inference

class TrainableSRFrozenSegPipeline(BasePipeline):
    '''
    Loss: Segmentation Loss
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

            seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}_fold{fold_idx}.pth')
            load_model_for_inference(model=seg_model, saving_name=seg_path)

            trainable_sr_frozen_seg_model = MultiStageModel(
                sr_model, 
                seg_model, 
                freeze_stage_1=False, 
                freeze_stage_2=True
            )
            optimizer = self._get_optimizer(trainable_sr_frozen_seg_model.parameters(), version)
            scheduler = self._get_scheduler(optimizer)
            gradient_clipper = self._get_gradient_clipper(trainable_sr_frozen_seg_model)

            fold_adapted_path = self._adapt_saving_path(fold_idx)

            trainer = MultiStageTrainer(
                config=self.config,
                model=trainable_sr_frozen_seg_model,
                device=self.device,
                train_loader=train_loader,
                validation_loader=validation_loader,
                criterion=criterion,
                validation_metrics=validation_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clipper=gradient_clipper,
                logger=logger,
                saving_name=fold_adapted_path
            )

            _, validation_results, training_time = trainer.train(epochs=self.config.epochs)
            fold_validation_results.append({
                "fold": fold_idx + 1,
                "fold_metrics": validation_results,
                "training_time": training_time
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

            trainable_sr_frozen_seg_model = MultiStageModel(
                sr_model, 
                seg_model, 
                freeze_stage_1=True, 
                freeze_stage_2=True
            )

            fold_adapted_path = self._adapt_saving_path(fold_idx)
            load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=fold_adapted_path)

            trainer = MultiStageTrainer(
                config=self.config,
                model=trainable_sr_frozen_seg_model,
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

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)
        trainable_sr_frozen_seg_model.to(self.device).eval()

        _, hr_mask, lr_image, _ = input_tensor

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = trainable_sr_frozen_seg_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1)
            predicted_mask = output_mask.squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, hr_mask)

            return {
                'input_image': lr_image,
                'target_mask': hr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }

    # PENDING TO DELETE
    def predict_random(self, dataset):
        sr_model = self._init_rcan()
        seg_model = self._init_unet()

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)
        trainable_sr_frozen_seg_model.to(self.device).eval()

        idx = random.randint(0, len(dataset) - 1)
        _, hr_mask, lr_image, _ = dataset[idx]

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = trainable_sr_frozen_seg_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1)
            predicted_mask = output_mask.squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, hr_mask)

            return {
                'input_image': lr_image,
                'target_mask': hr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }