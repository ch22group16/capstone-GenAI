# Email Subject Line Generation

## Problem Description
The task is to generate concise email subject lines from the body of the email. This involves identifying the most salient sentences and summarizing them into a few words.

## Dataset
We have used the dataset from [Github AESLC.](https://github.com/ryanzhumich/AESLC)

The dataset is annotated Enron Subject Line Corpus. It has dev, test, and train email datasets.
* Cleaned, filtered, and deduplicated emails from the Enron Email Corpus. 
* Sizes of train/dev/test splits: 14,436 / 1,960 / 1,906
* Average email length: 75 words
* Average subject length: 4 words

Dev and test datasets have @subject, @ann0,@ann1, and @ann2 to represent the subject lines of the mail.   
The train dataset has @subject as the subject line. 

# Data preprocessing

We have considered the below points to preprocess the data. 
* Removed unnecessary spaces with single space.
* Removed non-word and non-space characters.
* Extracted the content before @subject as email_body
* Extracted @subject as subject_line, @ann0 as subject_line1, @ann1 as subject_line2, and @ann2 as subject_line3.
* Created a single data file for dev, test, and train data with columns email_body, subject_line, subject_line1, subject_line2, subject_line3 contents.

We have analyzed the email body. The average email body has 2500 characters.  
To train the dataset, we have only considered the first 2500 characters of the email body from each email.

# Zero Shot Learning
Zero-shot learning (ZSL) is a machine learning scenario in which an AI model is trained to recognize and categorize objects   
or concepts without having seen any examples of those categories or concepts beforehand.
We have applied zero shot learning from test data. The result was not accurate. 

# Model Training
We have used the below models to train and test the data.
* BART
* T5 small
* Distill GPT2
* GPT2

The comparison of the Rouge score of each model is 
| Model  | Rouge 1 | Rouge 2 |  Rouge L  |
|----------|----------|----------|----------|
| BART    | 0.3006580658920587   | 0.15170335761446782   |0.2884998344468854   |
| T5 Small    | 0.27557781919299007   | 0.13223830150238886   |0.26705263028546283   |
| Distill GPT2    | Data 3   | Data 4   |Data 4   |
| GPT2    | Data 3   | Data 4   |Data 4   |
