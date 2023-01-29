# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:55:07 2023

@author: lomp
"""
__version__ = '0.1'

from RGP.recursiveGP import recursiveGP
from localGPR.localGPR import localGPR
from dataset import dataset
import argparse



def testsuite(rgpmodel, dataset_nr, options, num_coordinates=3, load=True, iterations=100, batch=10, savePDF=False, SCALING=1000):
    """
    rgpmodel : (class) RGP-model, i.e. recursiveGP, localGPR
    dataset_nr : (int) dataset
    options : (dict) options for each model 
    num_coordinates : (int) number of outputs coordinates, optional default 3 (xyz)        
    load : (bool) if True then the pre-splitted test/train data will be loaded
    iterations : (int), optional,  number of iterations default 100
    batch : (int), optional, batchsize, default 10
    SCALING : (int), optional, scale from meter to mm, default 1000        
    
    Returns: Will write PDF of plot to file output/dataset<dataset_nr>_<rgpmodel>.pdf
    usage:
    testsuite(recursiveGP, 6, options={"num_base_vectors" : 40, "strategy" : "JB"}, num_coordinates=3, load=True)

    """    
    
    classname = rgpmodel.__name__
    print("---- TESTSUITE for model "+classname+ " ----")
    
    print("Load dataset ", dataset_nr)
    dataobj = dataset(dataset_nr, classname)
    dataobj.load_dataframe(load=load)

    print("Create ", num_coordinates, " models of class ", classname)
    models = [rgpmodel(variance = dataobj.Variances[i], lengthscales = dataobj.Lengthscales[i], sigma=1E-05) for i in range(num_coordinates)]

    print("Initialitze ", num_coordinates, " models of class ", classname)
    for m in models:
        options["i"]=models.index(m)
        m.initialise(dataobj, options)

    print("Training of ", num_coordinates, " models of class ", classname)
    for m in models:
        j=models.index(m)
        m.train_batch(dataobj.XTrain, dataobj.YTrain[:,j:j+1],batch)

    print("Calculate Predictions and analyze error")   
    predictions = [m.predict(dataobj.XTest)[0] for m in models]
    difference, totalMSE, componentwiseErrors = dataobj.analyze_error(predictions,SCALING)
    
    print("Print error analysis")   
    dataobj.print_analysis(difference.T, totalMSE, componentwiseErrors, savePDF=savePDF)  
    print("---- END ----")


#---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='testsuite',
                                      description='Runs a test on datasets for the recursiveGP or local GPR model.',
                                      epilog="implemented by Christian Lomp")

    parser.add_argument('-d', '--dataset', help='indicates the dataset number', default="7")
    
    parser.add_argument('-m', '--model', help='indicates the model recursiveGP or localGPR', default="recursiveGP")
                        
    args = parser.parse_args()

    dataset_nr = int(args.dataset)

    if(args.model=="recursiveGP"):
        testsuite(recursiveGP, dataset_nr, options={"num_base_vectors" : 40, "strategy" : "JB"}, num_coordinates=3, load=True)
    elif(args.model=="localRGP"):
        testsuite(localGPR, dataset_nr, options={"wgen":0.8}, num_coordinates=3, load=True)
   	else:
   		print("Choose recursiveGP or localRGP")
