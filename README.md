# Sentiment Analysis Challenge

The aim of this challenge was to boost the accuracy of a sentiment analysis model for customer feedback. Given the small annotated dataset of 300 rows, I focused on utilizing two innovative techniques for sentiment analysis with limited data: sequential transfer learning and zero/few shot classification using an LLM.

I evaluated the performance of three fine-tuned transformers from HuggingFace Hub: `cardiffnlp/twitter-roberta-base-sentiment-latest`, `Seethal/sentiment_analysis_generic_dataset`, and `j-hartmann/sentiment-roberta-large-english-3-classes`. These models were chosen as they support multi-class sentiment analysis and represent state-of-the-art in fine-tuned transformers

Additionally, I considered the use of SetFit, a recent method for fine-tuning transformers on small datasets. SetFit uses contrastive learning to extract the maximum information from available data, achieving results comparable to fine-tuning RoBERTa Large on the full 3k example training set.

Finally, I explored the potential of zero-shot and few-shot classification using OpenAI's GPT-3 (`text-davinci-003`). The advantage of using this model is that it can achieve impressive results in text classification with the right prompts, without the need for any training data.

I went ahead with an 80/20 train/test split of the data extracted from sentiment_annotations.csv.

Here are the results: