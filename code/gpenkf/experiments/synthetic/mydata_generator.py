import numpy as np

from gpenkf.experiments.data_provider import DataProvider


class myDataGenerator(DataProvider):

    def __init__(self, dataset, sample_size, output_coord):
        self.XTrain        = dataset.XTrain
        self.YTrain        = dataset.YTrain[:,output_coord]#:output_coord+1]
        self.x_validation  = dataset.XTest
        self.f_validation  = dataset.YTest[ :,output_coord]#:output_coord+1]
        self.sample_size   = sample_size
        self.dataset       = dataset
        self.output_coord  = output_coord
        
    def generate_sample(self):
        return self.generate_sample_of_size(self.sample_size)

    def generate_sample_of_size(self, input_sample_size):
        I     = np.random.randint(0, self.XTrain.shape[0]-1, size=input_sample_size)
        x_new = self.XTrain[I,:self.XTrain.shape[1]]            
        f_new = self.YTrain[I]            
        #f_new_noised = f_new + np.random.normal(loc=0., scale=self.noise, size=(input_sample_size,))

        return x_new, f_new
    
    def get_error_mean(self, Ypred):
        difference = self.dataset.get_difference(Ypred, self.output_coord)
        return(np.mean(difference))
    
    
        