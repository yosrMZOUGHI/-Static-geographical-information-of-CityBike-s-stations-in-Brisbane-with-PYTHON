
# coding: utf-8

import sys, os
import json

import linecache
import logging
import argparse

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


"""
Please read README.md before launching 
"""



def GetExceptionInfo():
    # necessary to handle exceptions
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return (filename, lineno, line.strip(), exc_obj)





def init_logger(logger_id, logfilepath = None):
    # create logger
    logger = logging.getLogger(logger_id)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)
    if logfilepath is not None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add formatter to console handler
    ch.setFormatter(formatter)
    # add the console handler to the logger
    logger.addHandler(ch)
    return logger





def read_data( input_path):
    input_df= pd.DataFrame()
    logger.info("read_data starting.")
    try:
        # Importing the dataset
        with open(input_path) as json_data:
            input_json_data = json.load(json_data)
            # from json data to pandas dataframe
            input_df = pd.DataFrame.from_dict(input_json_data, orient='columns')
    except Exception as ex:
        filename, lineno, linetxt, exc_obj = GetExceptionInfo()
        tmpl = "Exception occurred: type {0}. line {1} in file {2}. Arguments:\n{3!r}."
        msg = tmpl.format(type(ex).__name__, lineno, filename, ex.args)
        logger.exception(msg)
    finally:
        return(input_df)
        logger.info("read_data ending.")
    
    





def apply_kmeans(input_df,n_clusters=3):
    logger.info("read_data starting.")
    try:
        assert n_clusters >1 
        #get essential featuresvalues
        features = np.array(list(zip(input_df["latitude"].values,input_df["longitude"].values)))
        # Number of clusters
        kmeans = KMeans(init='k-means++',n_clusters= n_clusters)
        # Fitting the input data
        kmeans = kmeans.fit(features)
        # Getting the cluster labels
        labels = kmeans.predict(features)
        # Centroid values
        centroids = np.array(kmeans.cluster_centers_)
    except Exception as ex:
        filename, lineno, linetxt, exc_obj = GetExceptionInfo()
        tmpl = "Exception occurred: type {0}. line {1} in file {2}. Arguments:\n{3!r}."
        msg = tmpl.format(type(ex).__name__, lineno, filename, ex.args)
        logger.exception(msg)
    finally:
        return(centroids, labels)
        logger.info("read_data ending.")
    





def plot_result(input_df,n_clusters,centroids , labels, output_path):
    
    logger.info("plot_result starting.")
    try:
        #plot result
        ptsymb = np.array(['b.','r.','m.','g.','c.','k.','b*','r*','m*','r^']);
        plt.figure(figsize=(12,12))
        plt.ylabel('Longitude', fontsize=12)
        plt.xlabel('Latitude', fontsize=12)
        for i in range(n_clusters):
            cluster=np.where(labels==i)[0]
            plt.plot(input_df.latitude[cluster].values,input_df.longitude[cluster].values,ptsymb[i])
        plt.scatter(centroids[:,0], centroids[:,1], marker="*", s=20)
        
    except Exception as ex:
        filename, lineno, linetxt, exc_obj = GetExceptionInfo()
        tmpl = "Exception occurred: type {0}. line {1} in file {2}. Arguments:\n{3!r}."
        msg = tmpl.format(type(ex).__name__, lineno, filename, ex.args)
        logger.exception(msg)
    finally:
        plt.savefig(output_path)
        logger.info("plot_result ending.")
    





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, help="path to the logfile",
                        default="./logging.log")
    parser.add_argument("--input", type=str, help="path to the inputfile",
                        default='./Brisbane_CityBike.json')
    parser.add_argument("--output", type=str, help="path to the outputfile",
                        default="./result.png")
    parser.add_argument("--kmeans", type=int, help="number of clusters",
                        default=3)


    args = parser.parse_args()
    logger = init_logger(".test_ds", args.logfile)
    input_path= args.input
    output_path= args.output
    n_clusters=args.kmeans
    #logger = init_logger(".test_ds", "./logging.log")
    logger.info("Starting.")
    try:
        
        assert os.path.exists(input_path)

        
        input_df=read_data(input_path)
        centroids,labels= apply_kmeans(input_df= input_df, n_clusters=n_clusters)        
        plot_result(input_df= input_df, 
                    n_clusters= n_clusters, 
                    centroids=centroids, 
                    labels=labels,
                    output_path=output_path)
    except Exception as ex:
        filename, lineno, linetxt, exc_obj = GetExceptionInfo()
        tmpl = "Exception occurred: type {0}. line {1} in file {2}. Arguments:\n{3!r}."
        msg = tmpl.format(type(ex).__name__, lineno, filename, ex.args)
        logger.exception(msg)
    finally:
        logger.info("Ending.")



