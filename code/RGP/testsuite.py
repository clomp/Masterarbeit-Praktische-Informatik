# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:55:07 2023

@author: lomp
"""
__version__ = '0.1'

from recursiveGP import recursiveGP
from dataset import dataset



def testsuite(rgpmodel, dataset_nr, options, num_coordinates, reuse=True, iterations=100, batch=10, SCALING=1000):
    
    classname = rgpmodel.__name__
    print("---- TESTSUITE for model "+classname+ " ----")
    
    print("Load dataset ", dataset_nr)
    dataobj = dataset(dataset_nr, classname)
    dataobj.load_dataframe(reuse)

    print("Create ", num_coordinates, " models of class ", classname)
    models = [rgpmodel(variance = dataobj.Variances[i], lengthscales = dataobj.Lengthscales[i], sigma=1E-05) for i in range(num_coordinates)]

    print("Initialitze ", num_coordinates, " models of class ", classname)
    for m in models:
        m.initialise(dataobj, options)

    print("Training of ", num_coordinates, " models of class ", classname)
    for m in models:
        j=models.index(m)
        m.train_batch(dataobj.XTrain, dataobj.YTrain[:,j:j+1],batch)

    print("Calculate Predictions and analyze error")   
    predictions = [m.predict(dataobj.XTest)[0].numpy() for m in models]
    difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    
    print("Print error analysis")   
    dataobj.print_analysis(difference.T, totalMSE, componentwiseErrors)  
    print("---- END ----")


testsuite(recursiveGP, 6, options={"num_base_vectors" : 40, "strategy" : "JB"}, num_coordinates=3, reuse=True)
