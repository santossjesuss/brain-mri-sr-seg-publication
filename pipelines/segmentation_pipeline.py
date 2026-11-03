import random
import torch
from torch.utils.data import Subset
from pipelines.base_pipeline import BasePipeline
from trainers.segmentation_trainer import SegmentationTrainer
from enums.hyperparameter_enum import HyperparameterVersion
from enums.resolution_enum import Resolution
from utils.model_persistence import load_model_for_inference

class SegmentationPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_hyperparameter_version(self):
        if self.data_resolution == Resolution.LR:
            version = HyperparameterVersion.V3
        else:   # HR or INTERPOLATED_HR
            version = HyperparameterVersion.V1

        return version

    def run(self, full_train_dataset, folds):
        version = self._get_hyperparameter_version()
        fold_validation_results = []

        for fold_idx, (train_indices, validation_indices) in enumerate(folds):
            print(f"\nFold {fold_idx + 1}/{self.config.num_folds}")

            train_dataset = Subset(full_train_dataset, train_indices)
            validation_dataset = Subset(full_train_dataset, validation_indices)

            train_loader = self._get_dataloader(train_dataset)
            validation_loader = self._get_dataloader(validation_dataset)

            model = self._init_unet()
            criterion = self._get_seg_loss(version=version)
            validation_metrics = self._get_seg_validation_metrics()
            optimizer = self._get_optimizer(model.parameters(), version)
            scheduler = self._get_scheduler(optimizer)
            logger = self._get_logger()

            fold_adapted_path = self._adapt_saving_path(fold_idx)

            trainer = SegmentationTrainer(
                config=self.config,
                model=model,
                device=self.device,
                train_loader=train_loader,
                validation_loader=validation_loader,
                data_resolution=self.data_resolution,
                criterion=criterion,
                validation_metrics=validation_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                saving_name=fold_adapted_path
            )

            _, validation_results = trainer.train(epochs=self.config.epochs)
            fold_validation_results.append({
                "fold": fold_idx + 1,
                "fold_metrics": validation_results
            })

        return fold_validation_results
    
    def test(self, test_dataset):
        fold_test_results = []

        for fold_idx in self.config.num_folds:
            print(f"\nFold {fold_idx + 1}/{self.config.num_folds}")

            test_loader = self._get_dataloader(test_dataset)
            
            model = self._init_unet()
            validation_metrics = self._get_seg_validation_metrics()

            load_model_for_inference(model, self.saving_path, fold_idx)

            trainer = SegmentationTrainer(
                config=self.config,
                model=model,
                device=self.device,
                validation_metrics=validation_metrics,
                data_resolution=self.data_resolution
            )

            test_metrics = trainer.test(test_loader)
            fold_test_results.append({
                "fold": fold_idx + 1,
                "fold_metrics": test_metrics
            })

        return fold_test_results

    # TO USE: FIX THE ARGUMENT validation_dataset 
    # def test_validation_subset(self, validation_dataset):
    #     validation_loader = self._get_dataloader(validation_dataset)

    #     model = self._init_unet()
    #     validation_metrics = self._get_seg_validation_metrics()

    #     load_model_for_inference(model, self.saving_path)

    #     trainer = SegmentationTrainer(
    #         config=self.config,
    #         model=model,
    #         device=self.device,
    #         validation_metrics=validation_metrics,
    #         data_resolution=self.data_resolution
    #     )

    #     return trainer.test(validation_loader)
    
    # PENDING TO DELETE
    def predict(self, input_tensor):
        model = self._init_unet()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        hr_image, hr_mask, lr_image, lr_mask = input_tensor

        if self.data_resolution == Resolution.HR:
            input_image = hr_image
            target_mask = hr_mask
        else:
            input_image = lr_image
            target_mask = lr_mask

        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask = model(input_image)
            predicted_mask = torch.argmax(output_mask, dim=1).squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, target_mask)

            if self.data_resolution == Resolution.HR:
                return {
                    'input_image': hr_image,
                    'target_mask': hr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }
            else:
                return {
                    'input_image': lr_image,
                    'target_mask': lr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }

    # PENDING TO DELETE
    def predict_random(self, dataset):
        model = self._init_unet()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()
        
        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]

        if self.data_resolution == Resolution.HR:
            input_image = hr_image
        else:
            input_image = lr_image

        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask = model(input_image)
            predicted_mask = torch.argmax(output_mask, dim=1).squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, lr_mask)

            if self.data_resolution == Resolution.HR:
                return {
                    'input_image': hr_image,
                    'target_mask': hr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }
            else:
                return {
                    'input_image': lr_image,
                    'target_mask': lr_mask,
                    'predicted_mask': predicted_mask,
                    'dice': dice
                }