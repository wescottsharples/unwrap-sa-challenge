"""Template definitions to be used in LLM prompts."""

zero_shot_template = """
Decide whether this product review's sentiment is positive, neutral, or negative. Output the sentiment as a single word.
For example, if the review is positive, output "Sentiment: positive".

Review: {text}
Sentiment:
"""

zero_shot_template_with_title = """
Decide whether this product review's sentiment is positive, neutral, or negative. Output the sentiment as a single word.
For example, if the review is positive, output "Sentiment: positive".

Title: {title}
Review: {text}
Sentiment:
"""

few_shot_template = """
Classify the following reviews as expressing positive, neutral, or negative sentiment.

Reviews:
Review #1
Title: "Who is Eligible Family Members for Hinge Health program?"
Text: "Just curious if anyone knows, who is considered an eligible family member to sign up for the Hinge Health benefit?"

Review #2
Title: None
Text: "Its way way much better than the other ride..., lol I love my lyft ride its more economical"

Review #3
Title: None
Text: "When I picked car it was 25 $ when my receipt came it was 50.99"

Review #4
Title: "when i click any link, it opens in a new tab. How do i make it open in the current tab, instead?"
Text: "according to what i found on Youtube, it appears this option used to exist but....they took it away???"

Review #5
Title: "Secret to getting more hours"
Text: "Start going to college. With this one simple trick you will watch your scheduled hours jump and still have your ETL calling."

Review #6
Title: {title}
Text: {text}

Sentiments (positive, negative, neutral):
1: neutral
2: positive
3: negative
4: negative
5: neutral
6:
"""
