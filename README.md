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

Dev and test datasets has @subject, @ann0,@ann1,@ann2 to represent subject lines of the mail.   
Train dataset has @subject to represent subject line. 

# Data preprocessing

We have considered below points to preprocess the data. 
* Removed unnecessary spaces with single space.
* Removed non-word and non-space character.
* Extracted the content before @subject as email_body
* Extracted @subject as subject_line, @ann0 as subject_line1, @ann1 as subject_line2, and @ann2 as subject_line3.
* Created a single data file for dev, test, and train data with columns as email_body, subject_line, subject_line1, subject_line2, subject_line3 contents.

We have analyzed the email body. The average email body has 2500 characters.  
To train the dataset, we have considered first 2500 characters of email body only from each email.

# Model Training

