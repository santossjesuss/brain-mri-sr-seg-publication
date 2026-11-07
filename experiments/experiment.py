class Experiment:
    def __init__(self, name, pipeline, config, folds, full_train_dataset, test_dataset, dataset_name, data_resolution=None):
        self.name = name
        self.pipeline = pipeline
        self.config = config
        self.folds = folds
        self.full_train_dataset = full_train_dataset
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.data_resolution = data_resolution

    def run(self):
        print(f"Starting experiment: {self.name}")

        pipeline = self.pipeline(config=self.config, experiment_name=self.name, dataset_name=self.dataset_name, data_resolution=self.data_resolution)
        fold_results = pipeline.run(self.full_train_dataset, self.folds)
        return fold_results

    def test(self):
        print(f"Testing experiment: {self.name}")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.test(self.test_dataset)
    
    # def test_validation_dataset(self):
    #     print(f"Testing experiment {self.name} on 'validation dataset'")

    #     pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
    #     return pipeline.test_validation_dataset(self.validation_dataset)

    def predict(self, idx):
        print(f"Predicting on input image using experiment '{self.name}'")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.predict(self.test_dataset[idx])
    
    def predict_random(self):
        print(f"Predicting random sample from dataset '{self.dataset_name}' using experiment '{self.name}'")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.predict_random(self.test_dataset)