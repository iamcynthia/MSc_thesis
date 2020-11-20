Abstract
This project applies three machine learning algorithms, support vector machine (SVM), K- nearest neighbour (KNN) and naïve Bayes (NB) for classifying different activities, compare the performance among them and improve these classifiers by under-sampling. SisFall dataset is used here for testing the classifiers. Data has 
been denoised by Butterworth low pass filter and segmented the pre-processed data into 5 seconds a chunk. Each chunk has been summarized into 12 features and 
feature extraction techniques will be implemented to reduce dimensionality. There are three versions of dataset after segmentation which are based on different 
methods for the given labels. Radial basis function (RBF) SVM is the best classifier in binary version of dataset, which is 89.41% on accuracy score. In 9 classes 
and 13 classes version of dataset, KNN obtains the best accuracy score, which are 67.98% and 60.5% respectively. Moreover, comparing the results after addressing 
data imbalanced problem, though the accuracy score decreased, the sensitivity and specificity score increased when using one-sided selection (OSS) and synthetic 
minority oversampling technique (SMOTE) + Tomek link (T-Link). In conclusion, KNN may be the most suitable algorithms for classifying gait patterns and the 
improvement of the classifier implies that label imbalanced is one of the main issues.
-----------------------------------------------------------------------------

1. step1_data_preprocessing.py
---- including convert the original datasets' txt filte to csv file, plot gait patterns for each subjects, 3 denoising methods

2. step2_Segmentation.py
---- segment dataset into several 5 seconds chunk and give labels 

3. 9Classes.py
---- the process and results for 9 classes version of datset

4. 2Classes.py
---- the process and results for 2 classes version of datset

5. 13Classes.py
---- the process and results for 13 classes version of datset

6. Activities.csv
---- includes the activities information (i.e. code, description, trial, duration)

7. <File> SisFall_dataset
---- the public dataset

8. statement.txt
---- statement certifying the work as my own
