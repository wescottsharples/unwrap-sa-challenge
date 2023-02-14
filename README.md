# Sentiment Analysis Challenge

The aim of this challenge was to boost the accuracy of a sentiment analysis model for customer feedback. Given the small annotated dataset of 300 rows, I focused on utilizing two innovative techniques for sentiment analysis with limited data: sequential transfer learning and zero/few shot classification using an LLM.  

I evaluated the performance of three fine-tuned transformers from HuggingFace Hub: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [`Seethal/sentiment_analysis_generic_dataset`](https://huggingface.co/Seethal/sentiment_analysis_generic_dataset?text=I+like+you.+I+love+you), and [`j-hartmann/sentiment-roberta-large-english-3-classes`](https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes). These models were chosen as they support multi-class sentiment analysis and represent state-of-the-art in fine-tuned transformers

Additionally, I considered the use of [SetFit](https://arxiv.org/abs/2209.11055), a recent method for fine-tuning transformers on small datasets. SetFit uses contrastive learning to extract the maximum information from available data, achieving results comparable to fine-tuning RoBERTa Large on a training set with 3K examples.

Finally, I explored the potential of zero-shot and few-shot classification using OpenAI's [GPT-3](https://platform.openai.com/ai-text-classifier) (`text-davinci-003`). The advantage to this approach is that it can achieve impressive results with zero training examples.

I went ahead with an 80/20 train/test split of the data extracted from [`sentiment_annotations.csv`](data/raw/sentiment_annotations.csv).

A summary of the results is shown below (`Accuracy` is given by the support-weighted mean of F1 scores per label):

| Model             | Accuracy | Inference Time |
| ----------------- | -------- | -------------- |
| Baseline          | 72%      |                |
| SetFit            | 80%      | 0.16s          |
| Hartmann          | 75%      | 1.17s          |
| GPT-3 (zero-shot) | 73%      |                |
| GPT-3 (few-shot)  | 72%      |                |
| Cardiff NLP       | 68%      | 1.4s           |
| Seethal           | 68%      | 0.31s          |

For a more full-flavored breakdown of model performance, please see [`results.txt`](results.txt).

I think the results speak for themselves. The fine-tuned SetFit model was the best performer both in terms of accuracy and inference time and would be my recommendation. I wanted to experiment with generating additional training examples using GPT-3 to further fine-tune SetFit, but ultimately ran out of time.

If you would like further insight into my thought process please see the [`notebooks`](notebooks). To reproduce the results on the latest data in MySQL, you can use the following:

```bash
python --version # ~3.10
mv .env.example .env # add OPENAI_API_KEY if you want to call GPT-3
pip install -r requirements.txt
pip install -e .
python scripts/compare_all_models.py
```

This will produce both a `results.txt` file for high-level results and a `data/results/results.pkl` file for the full results dictionary which can be loaded into a notebook for manual inspection.