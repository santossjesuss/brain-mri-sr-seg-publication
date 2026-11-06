import random
import torch
from torch.utils.data import Subset
from pipelines.base_pipeline import BasePipeline
from trainers.super_resolution_trainer import SuperResolutionTrainer
from enums.hyperparameter_enum import HyperparameterVersion
from utils.model_persistence import load_model_for_inference

class SuperResolutionPipeline(BasePipeline):
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

            model = self._init_rcan()
            criterion = self._get_sr_loss()
            validation_metrics = self._get_sr_validation_metrics()
            optimizer = self._get_optimizer(model.parameters(), version)
            scheduler = self._get_scheduler(optimizer)
            logger = self._get_logger()

            fold_adapted_path = self._adapt_saving_path(fold_idx)

            trainer = SuperResolutionTrainer(
                config=self.config,
                model=model,
                device=self.device,
                train_loader=train_loader,
                validation_loader=validation_loader,
                criterion=criterion,
                validation_metrics=validation_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                saving_name=fold_adapted_path,
                logger=logger
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

            model = self._init_rcan()
            validation_metrics = self._get_sr_validation_metrics()

            fold_adapted_path = self._adapt_saving_path(fold_idx)
            load_model_for_inference(model, fold_adapted_path)

            trainer = SuperResolutionTrainer(
                config=self.config,
                model=model,
                device=self.device,
                validation_metrics=validation_metrics
            )

            test_metrics = trainer.test(test_loader)
            fold_test_results.append({
                "fold": fold_idx + 1,
                "fold_results": test_metrics
            })

        return fold_test_results

    # PENDING TO DELETE
    def predict(self, input_tensor):
        model = self._init_rcan()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        hr_image, hr_mask, lr_image, lr_mask = input_tensor

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_image = model(input_image)
            predicted_image = output_image.squeeze(0).cpu()

            return {
                'input_image': lr_image,
                'target_image': hr_image,
                'predicted_image': predicted_image
            }

    # PENDING TO DELETE
    def predict_random(self, dataset):
        model = self._init_rcan()
        load_model_for_inference(model, self.saving_path)
        model.to(self.device).eval()

        idx = random.randint(0, len(dataset) - 1)
        hr_image, hr_mask, lr_image, lr_mask = dataset[idx]

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_image = model(input_image)
            predicted_image = output_image.squeeze(0).cpu()

            return {
                'input_image': lr_image,
                'target_image': hr_image,
                'predicted_image': predicted_image
            }