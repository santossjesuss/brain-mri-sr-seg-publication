import os
import random
import torch
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference

class FrozenSRFrozenSegPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        print('This pipeline only supports testing.')
        print('It has already both models trained.')

    def test(self, test_dataset):
        fold_test_results = []

        for fold_idx in self.config.num_folds:
            test_loader = self._get_dataloader(test_dataset)

            sr_model = self._init_rcan()
            seg_model = self._init_unet()
            validation_metrics = self._get_seg_validation_metrics()

            sr_path = os.path.join(self.config.folder_name, f'{self.config.sr_name}_fold{fold_idx}.pth')
            seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}_fold{fold_idx}.pth')
            load_model_for_inference(sr_model, sr_path)
            load_model_for_inference(seg_model, seg_path)

            sr_seg_model = MultiStageModel(
                sr_model, 
                seg_model, 
                freeze_stage_1=True, 
                freeze_stage_2=True
            )

            trainer = MultiStageTrainer(
                config=self.config,
                model=sr_seg_model,
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
        sr_path = os.path.join(self.config.folder_name, f'{self.config.sr_name}.pth')
        seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}.pth')

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        load_model_for_inference(sr_model, sr_path)
        load_model_for_inference(seg_model, seg_path)

        sr_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        ).to(self.device)
        sr_seg_model.eval()

        _, hr_mask, lr_image, _ = input_tensor

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = sr_seg_model(input_image)
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
        sr_path = os.path.join(self.config.folder_name, f'{self.config.sr_name}.pth')
        seg_path = os.path.join(self.config.folder_name, f'{self.config.hr_seg_name}.pth')

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        load_model_for_inference(sr_model, sr_path)
        load_model_for_inference(seg_model, seg_path)

        sr_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        ).to(self.device)
        sr_seg_model.eval()

        idx = random.randint(0, len(dataset) - 1)
        _, hr_mask, lr_image, _ = dataset[idx]

        input_image = lr_image
        input_image = input_image.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output_mask, _ = sr_seg_model(input_image)
            output_mask = torch.argmax(output_mask, dim=1)
            predicted_mask = output_mask.squeeze(0).cpu()
            dice = self._compute_dice(predicted_mask, hr_mask)

            return {
                'input_image': lr_image,
                'target_mask': hr_mask,
                'predicted_mask': predicted_mask,
                'dice': dice
            }