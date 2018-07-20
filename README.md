# Static geographical information of CityBike‘s stations in Brisbane
This is a clustering analytics application to perform a Kmeans clustering based on the location of the bike stations. This application is build with python3.

# Python libraries
Make sure you have the latest versions of pandas, sklearn and numpy.

# Logging
Exceptions and errors are trackable in the logging file.

# Launching 
From UNIX command line, use one of the following lines:

1. default configuration ( input file, output file and looging file are in the same directory as the source code, number of cluster is equal to  3 by default)

  ```
python3 <path_to_the_source_code>/kmeansclustering.py
  ```

2. Configure input path or output path or logging path or number of clusters (or all of that)

  ```
python3 <path_to_the_source_code>/kmeansclustering.py --input=<inputpath> --output=<outputpath> --logfile=<logfilepath> --kmeans=<nbr_of_clusters>
  ```
# Input
input must be a json format. We didn't verify the format in the code. We used the "latitude", "longitude" values and we supposed it is always provided with these names.

# Output
output is a plot: the dots represent the clusters ( each color is a different cluster) and the black stars are kmeans centroids.

# About kmeans
K-means is  popular for cluster analysis in data mining.In fact, K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). 
The results of the K-means clustering algorithm are:

1. Centroids of the K clusters, which can be used to label new data
2. Labels for the training data (each data point is assigned to a single cluster)


# Author
Yosr MZOUGHI
yosr.mzoughi@gmail.com
+33(0)629988364
# -Static-geographical-information-of-CityBike-s-stations-in-Brisbane-with-PYTHON
