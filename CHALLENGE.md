# Unrap.ai Challenge

## Goal

At Unwrap, our job is to help our customers make sense of their user feedback. One way we do this is by aggregating all of their raw feedback and returning each feedback entry’s sentiment. This allows our customers to know if overall feedback is becoming more positive or negative, but more importantly allows them to analyze the sentiment of particular product features and monitor changes over time.

Your task is to access feedback entries from a MySQL database, test multiple approaches for sentiment analysis, compare against and improve upon Unwrap’s existing sentiment accuracy, and explain your findings in a written statement.

### Specifically, the requirements are:

1. Connect to a MySQL database, and retrieve the necessary data
2. Test various approaches to improve our sentiment model’s accuracy
3. Select the optimal approach based on data-driven analysis comparing to Unwrap’s base sentiment model
4. Write a written statement explaining your optimal approach, and what led to that decision

### Things we’re looking for:

- Thorough analysis of multiple approaches to improve the sentiment model’s accuracy
- An improvement to our current model’s accuracy
- A well-written, defensible written statement explaining your approach and your recommendation
- Well organized, readable Python code

### Things you don’t need to worry about: 

- Deploying/hosting your sentiment model(s)
- The particular approach you choose. We care most about the performance of your approach, not necessarily how you achieve it

## Accessing the data

At a high level, Unwrap has feedback entries (e.g. think a Yelp review or app store review), which are each made up of 1 or more sentences.

You have one table in your MySQL database:
- **`feedback_entry`**
	- Each row is a feedback entry
	- You’ll notice that each entry has a title (occasionally), the entry text, its current sentiment value, and other metadata.

You’ll need to programmatically read from the `feedback_entry` table to perform your analysis. Note that you should read from the database each time you run your analysis, as opposed to
performing a one-time export from the database since you have to assume the database will update periodically.

## Sentiment Analysis

The primary goal of this exercise is to increase the accuracy of our sentiment model. For each entry, the current sentiment score is stored in the database in the `sentiment` column. Please see the attached `sentiment_annotations.csv` document, which will allow you to measure your 
model’s performance against the ground-truth annotations. In order to improve the sentiment model’s accuracy, you can try various techniques like using one or many open source models, building your own classifier, or any combination of these approaches.

## Written Statement

Produce a written summary of the approaches you took, reasons for why you made certain decisions, and your recommendation for which approach to implement.

## Scoring

| **Area**           | Aspect                                | Points |
| ------------------ | ------------------------------------- | ------ |
| Database           | Successfully accessing DB             | 2      |
| Sentiment Analysis | Analytical testing of various options | 5      |
| Sentiment Analysis | Improving accuracy score              | 5      |
| Written Statement  | Logical & well-written statement      | 5      |
| General            | Clean and clear code                  | 5      |

# Challenge Solution

## High-level Approach

To begin this challenge I thought about