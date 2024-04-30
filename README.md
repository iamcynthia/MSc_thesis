# Gait Patterns Classification
Ageing society has become a normal situation nowadays, and falls become a public health concern simultaneously and lead to devastating consequences for elders, such as morbidity, mortality, and loss of independence. 

SisFall (Sucerquia, López, and Vargas-Bonilla 2017, 198) is the public dataset we used in this project. It includes 19 activities of daily living (ADLs) and 15 falls performed by 23 young adults and 15 elders.

## Aims and Objectives
This project aims to classify activities of daily living (ADLs) and different fall activities by several supervised learning algorithms. In addition, it is to find out which kind of classifier has the highest accuracy on classifying all the activities.

- Understand the gait pattern for each activity
- Implement three filtering methods to denoise raw data and choose the best one
- Segment the denoising data and give a different version of labels (i.e. binary and multi-class)
- Extract the best features for the data by principal components analysis (PCA) and linear discrimination analysis (LDA)
- Test three machine learning algorithms for each version of data and compare their efficiency
- Improve the best classifier through under-sampling the data


## Evaluation
### Binary Version (fall v.s. no-fall)
| Dataset Version    | Classifier              | Accuracy Score | Sensitivity Score | Specificity Score |
|--------------------|-------------------------|----------------|------------------|-------------------|
| Binary             | RBF SVM                 | 89.41%         | -                | -                 |
| Binary (Balanced)  | RBF SVM (OSS)           | 88.73%      | Increased        | Increased         |
| Binary (Balanced)  | RBF SVM (SMOTE + T-Link)| 88.43%     | Increased        | Increased         |

In binary version, most improving techniques do not have a significant effect.

### 9 Classes Version
| Dataset Version    | Classifier              | Accuracy Score | Sensitivity Score | Specificity Score |
|--------------------|-------------------------|----------------|------------------|-------------------|
| 9 Classes          | KNN                     | 67.98%         | -                | -                 |
| 9 Classes (Balanced) | KNN (OSS)             | 57.56%      | Increased        | Increased         |
| 9 Classes (Balanced) | KNN (SMOTE + T-Link)  | 56.49%      | Increased        | Increased         |

For the dataset with 9 classes, the technique, SMOTE + T-Link, with both randomly oversampling and under-sampling is useful.

### Multiclass Version
| Dataset Version    | Classifier              | Accuracy Score | Sensitivity Score | Specificity Score |
|--------------------|-------------------------|----------------|------------------|-------------------|
| 13 Classes         | KNN                     | 60.5%          | -                | -                 |
| 13 Classes (Balanced) | KNN (OSS)            | 49.23%      | Increased        | Increased         |
| 13 Classes (Balanced) | KNN (SMOTE + T-Link) | 48.54%      | Increased        | Increased         |

The 13 classes version of dataset has some improvements when under-sampling the selected classes. OSS seems to have better improvement among all other methods when under-sampling the classes “fall forward” and “jogging”.


## Folder Structure
```
.
├── SisFall_dataset              # the public dataset
├── 2Classes.py                  # the process and results for 2 classes version of datset
├── 9Classes.py                  # the process and results for 9 classes version of datset
├── 13Classes.py                 # the process and results for 13 classes version of datset
├── Activities.csv               # includes the activities information (i.e. code, description, trial, duration)
├── FinalReport.pdf              # MSc Thesis
├── README.md               
├── statement.txt                # statement certifying the work as my own
├── step1_data_preprocessing.py  # including convert the original datasets' txt filte to csv file, plot gait patterns for each subjects, 3 denoising methods
├── step2_Segmentation.py        # egment dataset into several 5 seconds chunk and give labels
└── 
```
