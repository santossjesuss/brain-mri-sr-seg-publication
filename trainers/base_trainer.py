from abc import ABC, abstractmethod
from utils.gpu import synchronize_gpu
from utils.memory import reset_memory_statistics, query_peak_memory_allocation
from utils.time import format_elapsed_time
from utils.model_persistence import save_model_for_inference, load_model_for_inference
import time 

class BaseTrainer(ABC):
    def __init__(
            self, 
            config,
            model, 
            device,
            train_loader=None, 
            validation_loader=None, 
            criterion=None, 
            validation_metrics=None, 
            optimizer=None, 
            scheduler=None,
            gradient_clipper=None,
            saving_name=None,
            logger=None,
            img_logger=None
        ):
        super().__init__()
        self.config = config
        self.device = device
        self.patience = self.config.patience
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model.to(device)
        self.criterion = criterion.to(device) if criterion else None
        self.validation_metrics = validation_metrics.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clipper = gradient_clipper
        self.saving_name = saving_name
        self.logger = logger
        self.img_logger = img_logger

    def train(self, epochs):
        best_validation_score = float('-inf')
        early_stop_counter = 0

        reset_memory_statistics()
        synchronize_gpu()
        start_time = time.perf_counter()

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch, epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics[self.get_primary_metric_name()]
            self.scheduler.step(validation_score)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tValidation Metrics: {validation_metrics}')

            if self.logger:
                self.logger.log_metric("Loss", train_loss, epoch, phase="Training")
                self.logger.log_metrics(validation_metrics, epoch, phase="Validation")

            if validation_score > best_validation_score:
                early_stop_counter = 0
                best_validation_score = validation_score
                best_metrics = validation_metrics
                save_model_for_inference(self.model, self.saving_name)
            else:
                early_stop_counter += 1
                print(f'Early Stopping: {early_stop_counter}/{self.patience}')

                if early_stop_counter >= self.patience:
                    break
        
        synchronize_gpu()
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        training_time = format_elapsed_time(elapsed_time)
        peak_memory = query_peak_memory_allocation()
        
        load_model_for_inference(self.model, self.saving_name)
        return self.model, best_metrics, training_time, peak_memory

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        pass

    @abstractmethod
    def _evaluate(self, dataloader, description):
        pass

    def _validate(self):
        return self._evaluate(self.validation_loader, description='Validating')
    
    def test(self, test_loader):
        testing_metrics = self._evaluate(test_loader, description='Testing')
        if self.logger:
            self.logger.log_metrics(metrics_dict=testing_metrics, phase='Testing')

        print(f'\tTesting Metrics: {testing_metrics}')

        return testing_metrics
    
    @abstractmethod
    def _prepare_batch(self, batch):
        pass

    @abstractmethod
    def get_primary_metric_name(self):
        pass