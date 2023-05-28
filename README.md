# CNN Language Model for Toxic Languages Classification

## About this project

The project aims to build a language model that can accurately classify and categorize toxic comments on social media platforms. The model will be trained on a pre-labeled dataset containing toxic and non-toxic online comments to identify different levels of toxic on media platforms. </p>

In addition, the project aims to identify different categories of toxicity, such as identity-based hate, threats, insults, and others. </p>

The baseline model for this project is a simple Bag-Of-Words (BoW) model, and the main model is a Convolutional Neural Network (CNN). Both models use `GloVe` for word embedding.

## Dataset 

The training dataset for this project is Jigsaw Toxic Comment Classification Challenge dataset on Kaggle ([link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)). </p>

The average length of any individual comment’s raw text is 68, measured in the number of characters of the comment. The proportion of comments in the dataset that have over 60 words is 32%. The proportion of comments that have over 100 words is 18%. And the proportion of comments that have over 200 words is only 6%. </p>

For the purpose of training model, a re-sampled dataset was applied for this project. The dataset contains 28929 records, with half classified as `non-toxic` and the other half classified as `toxic`. The dataset is split into 70: 20: 10 for training, validation and testing.


## Models and Performance

### (1) Baseline Model: BOW

needs to fill up

### (2) CNN Model V0: 

The first model has 4 different filter sizes: 2, 3, 4 and 5, each focuses on different sizes of N-grams. We assigned 64 filters of each sizes for this model. The fully-connected neural network has 256 input features and binary outputs. The model's dropout rate is set to be 0.5 to avoid over fitting. </p>

The model was trained on MacBook Pro M2 CPU for 10 epochs, which took around 30 minutes.

```python

(CNN_NLP(
   (conv1d_list): ModuleList(
     (0): Conv1d(300, 64, kernel_size=(2,), stride=(1,))
     (1): Conv1d(300, 64, kernel_size=(3,), stride=(1,))
     (2): Conv1d(300, 64, kernel_size=(4,), stride=(1,))
     (3): Conv1d(300, 64, kernel_size=(5,), stride=(1,))
   )
   (fc): Linear(in_features=256, out_features=2, bias=True)
   (dropout): Dropout(p=0.5, inplace=False)
 )
```

#### PARAMETERS

|Description         |Values           |
|:------------------:|:---------------:|
|input word vectors  |GloVe            |
|embedding size      |300              |
|filter sizes        |(2, 3, 4, 5)     |
|num filters         |(64, 64, 64, 64) |
|activation          |ReLU             |
|pooling             |1-max pooling    |
|dropout rate        |0.5              |

#### PERFORMANCE

**With 10 epochs of training:**

For training and validation, we see the following performance: 

| Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
|:------:|:------------:|:----------:|:---------:|:-------:|
|   1    |   0.515071   |  0.387810  |   85.50   |  222.95 | 
|   2    |   0.357476   |  0.314949  |   87.84   |  214.95 |
|   3    |   0.307458   |  0.286911  |   88.58   |  217.30 |
|   4    |   0.283827   |  0.271694  |   89.29   |  212.41 |
|   5    |   0.269294   |  0.261957  |   89.29   |  217.79 |
|   6    |   0.257123   |  0.254316  |   89.50   |  238.22 |
|   7    |   0.247575   |  0.248300  |   89.76   |  223.09 |
|   8    |   0.239426   |  0.243885  |   89.81   |  216.53 |
|   9    |   0.233904   |  0.239794  |   90.08   |  210.70 |
|  10    |   0.227885   |  0.236559  |   90.17   |  221.15 |

![alternatvie text](https://github.com/yifu-hou/toxic-language-filter/blob/main/yifu/graphs/best_m0_10epochs_val_acc.png)

For testing, the first CNN model performed well on 2865 testing data: </p>

| Accuracy |	Precision |	Recall	 |  F1       |
|:--------:|:------------:|:--------:|:---------:|
| 0.914455 |	0.920194  |	0.910714 |	0.91543  |

#### FINDINGS IN MODEL PREDICTION

Checking the False Negative and False Positive predictions, we can see some patterns: </p>

**False Negatives** </p>
*(toxic comments that the model failed to identify)*: </p> 

- misspelled rarely-seen slurs

    - id: `e22a2557c33d5df3` 
    - comment: `tno thanks mate p i s s offe` 

    - id: `f16ec7cafd4ff73c` 
    - comment: `you obviously know shit-nothing about physics, if the buildings were ...` 

- mitigated or weirdly-worded insults 

    - id: `06a44c69b4c3fb43` 
    - comment: `In response to your recent comment on my talk page. I suggest you contract cancer.`

- ambiguous connotations for language models

    - id: `e8d66a843390f637` 
    - comment: `- Do it and I will cut you`


**False Positives** </p>
*(toxic comments that the model wrongly identified as toxic)*: </p> 

- triggering words used in non-toxic context

    - id: `96b055eca6cc56b0` 
    - comment: `a debate to see why he was so virulently anti-Zionist`

    - id: `289b9ebd8ee46b91` 
    - comment: `This article is useless without pics`

**Debatable Labels**

It is worth mentioning that the labeling of original dataset is not perfect - there are a few comments that might be wrongly labelled. </p>

id: `7ba73d7587ca3568` </p>
comment: </p>
```
I have noticed that underneath Sexuality on this page it says 'Gaydolf was a homsexual foggot' i think it would be appropriate to remove this if somebody can.   carf9
```
(labeled as `toxic`, our model identified as `not toxic`) </p>

id: `d4090f8db8939d73` </p>
comment: </p> 

```
Yo who the heeck wrote this and how the heck do they even know what happends.
```
(labeled as `toxic`, our model identified as `not toxic`) </p>

id: `583c3800a5b3b464` </p>
comment: </p> 
```
Do what you want, but you'll never get rid of me, that's a promise. Give your sister a kiss for me.
```
(labeled as `not toxic`, our model identified as `toxic`) </p>

**With 20 epochs of training:**

The exact same model have been trained for another 10 epochs. We see slight improvement in accuracy, recall and f1. However, the validation accuracy start to reach a plateau after around 15 epochs. </p>

Performance in training:

| Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
|:------:|:------------:|:----------:|:---------:|:-------:|
   1     |   0.518177   |  0.390717  |   85.49   |  206.55  
   2     |   0.356052   |  0.314333  |   87.69   |  213.67  
   3     |   0.305211   |  0.285256  |   88.55   |  216.24  
   4     |   0.282695   |  0.270456  |   89.24   |  276.86  
   5     |   0.267897   |  0.261121  |   89.39   |  263.94  
   6     |   0.258804   |  0.254165  |   89.70   |  269.75  
   7     |   0.247487   |  0.248454  |   90.01   |  231.39  
   8     |   0.237717   |  0.243787  |   90.10   |  206.41  
   9     |   0.233450   |  0.240119  |   90.24   |  205.30  
  10     |   0.227105   |  0.237166  |   90.38   |  205.57  
  11     |   0.220388   |  0.234342  |   90.53   |  205.35  
  12     |   0.215990   |  0.231839  |   90.53   |  205.72  
  13     |   0.212377   |  0.229920  |   90.63   |  210.60  
  14     |   0.207373   |  0.228286  |   90.65   |  208.69  
  15     |   0.204846   |  0.227199  |   90.86   |  209.43  
  16     |   0.198263   |  0.225998  |   90.84   |  207.99  
  17     |   0.195395   |  0.224842  |   91.01   |  207.73  
  18     |   0.189093   |  0.223940  |   90.89   |  208.72  
  19     |   0.188172   |  0.222998  |   91.00   |  203.57  
  20     |   0.184841   |  0.222217  |   91.17   |  351.67 

![alternatvie text](https://github.com/yifu-hou/toxic-language-filter/blob/main/yifu/graphs/best_m0_20epochs_loss.png)

![alternatvie text](https://github.com/yifu-hou/toxic-language-filter/blob/main/yifu/graphs/best_m0_20epochs_val_acc.png)

Performance in testing:

| Accuracy |	Precision |	Recall	 |  F1       |
|:--------:|:------------:|:--------:|:---------:|
| 0.92074  |	0.918882  |	0.925824 |	0.92234 |


### (3) CNN Model V1: 

#### PARAMETERS

|Description         |Values           |
|:------------------:|:---------------:|
|input word vectors  |GloVe            |
|embedding size      |300              |
|filter sizes        |(2, 3, 4, 5)     |
|num filters         |(128,128,128,128)|
|activation          |ReLU             |
|pooling             |1-max pooling    |
|dropout rate        |0.5              |

#### PERFORMANCE

**With 10 epochs of training:**

| Accuracy |	Precision |	Recall	 |  F1       |
|:--------:|:------------:|:--------:|:---------:|
| 0.915852 |	0.917526  |	0.916896 |	0.917211 |

**With 20 epochs of training:**

No data


### (3) CNN Model V1 (low DropOut rate): 

#### PARAMETERS

|Description         |Values           |
|:------------------:|:---------------:|
|input word vectors  |GloVe            |
|embedding size      |300              |
|filter sizes        |(2, 3, 4, 5)     |
|num filters         |(128,128,128,128)|
|activation          |ReLU             |
|pooling             |1-max pooling    |
|dropout rate        |0.2              |

#### PERFORMANCE

**With 10 epochs of training:**

| Accuracy |	Precision |	Recall	 |  F1       |
|:--------:|:------------:|:--------:|:---------:|
| 0.92074  |	0.918882  |	0.925824 |	0.92234  |
