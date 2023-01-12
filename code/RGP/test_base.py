# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:55:07 2023

@author: lomp
"""
__version__ = '0.1'

from recursiveGP import recursiveGP
from dataset import dataset



def testbase(rgpmodel, dataset_nr, options, num_coordinates=3, reuse=True, iterations=100, batch=10, SCALING=1000):
 
    
    classname = rgpmodel.__name__
    dataobj = dataset(dataset_nr, classname)
    dataobj.load_dataframe(reuse)    
    models = [rgpmodel(variance = dataobj.Variances[i], lengthscales = dataobj.Lengthscales[i], sigma=1E-05) for i in range(num_coordinates)]    
    for m in models:
        options["i"]=models.index(m)
        m.initialise(dataobj, options)
    
    for m in models:
        j=models.index(m)
        m.train_batch(dataobj.XTrain, dataobj.YTrain[:,j:j+1],batch)

    predictions = [m.predict(dataobj.XTest)[0] for m in models]
    difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    return(totalMSE)

#base nur 39