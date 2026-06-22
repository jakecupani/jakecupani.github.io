---
title: Using Reddit Comment Sentiment Analysis as a Proxy for COVID-19 Vaccine Acceptance by State  💉
publishDate: August 1st, 2022
description: Research project focused on Reddit sentiment analysis and vaccines.
tags: ['Data Science 📈']
---

# Can Reddit Sentiment Predict COVID-19 Vaccine Acceptance?

With the unprecedented spread of the COVID-19 pandemic, social media quickly transformed into a massive—and often highly divisive—forum for discussing the virus and the ensuing vaccines. While the vaccines proved highly effective at preventing severe illness, public hesitancy fueled by misinformation created major hurdles for public health.

This raises a fascinating question: Can we analyze this vast ocean of social media commentary to accurately measure real-world vaccine acceptance on a state-by-state level?

In my recent paper, *"Using Reddit Comment Sentiment Analysis as a Proxy for COVID-19 Vaccine Acceptance by State"*, I set out to determine if natural language processing (NLP) and sentiment analysis could serve as a reliable proxy for public behavior.

### Why Reddit?

When studying localized sentiment, not all social media platforms are created equal. Reddit offers a distinct advantage because its community-driven "subreddit" structure inherently categorizes discussions by topic and location. Instead of parsing through millions of random tweets with hashtags, researchers can look directly at state-specific subreddits to gather hyper-local data.

### The Data Collection

To test this theory, I used the Python Reddit API Wrapper (PRAW) to scrape top-level comments from state subreddits that included the keywords "covid" or "vaccine". The data was cleaned to remove deleted comments, special characters, and unhelpful formatting.

This resulted in two primary text datasets:

* **The Full Dataset:** Containing 22,824 unique comments spanning from the start of the pandemic (January 1st, 2020) to March 3rd, 2022.
* **The August Dataset:** A smaller subset of 7,061 comments dating from August 1st, 2021 to March 1st, 2022 (created specifically to manage API resource limits for the more advanced models).

This text data was then compared against public data from the CDC detailing the percentage of fully vaccinated individuals in each state.

### Three Approaches to Sentiment Analysis

Assigning a quantifiable "feeling" to text is notoriously difficult. To find the most accurate proxy for vaccine acceptance, I tested three different NLP methodologies:

**1. Basic TextBlob Polarity Scoring**
TextBlob is a widely used Python library for processing textual data. It evaluated both post titles and individual comments, assigning them a polarity score ranging from -1 (negative) to 1 (positive).

**2. Title and Comment Polarity Matching**
This ad-hoc method attempted to capture the nuance of "anti-vax" sentiment by comparing a comment against the headline it was responding to. For example, if a post featured a negative, anti-vaccine title, and a user replied with a positive polarity comment, the logic assumes the user *agrees* with the anti-vaccine sentiment. In these cases, the comment's polarity was mathematically inverted (multiplied by -1) to reflect the true stance.

**3. GPT-3 Polarity Scoring**
Leveraging the massive 175-billion parameter GPT-3 model from OpenAI, this technique fed the AI a simple plain-English prompt: *"Decide whether a comment's sentiment is positive, neutral, or negative"* followed by the text. These responses were then mapped to numeric values (1, 0, and -1).

### The Results: What Worked and What Failed

The correlation tests between the state-averaged sentiment scores and the CDC's average vaccination rates yielded incredibly revealing results:

* **TextBlob is highly effective on comments:** Using the standard TextBlob polarity scoring on the Full Dataset comments resulted in a strong positive correlation of roughly **0.50**. However, evaluating the post *titles* with TextBlob was much less effective, yielding a correlation of only **0.31**.
* **GPT-3 holds its own:** Tested on the smaller August Dataset, the GPT-3 model achieved a solid correlation coefficient of roughly **0.42**. This slightly outperformed TextBlob's score of **0.40** on that exact same timeframe.
* **The Matching Method fell flat:** The experimental Title and Comment Matching technique was largely unsuccessful, yielding very weak correlations ranging from just **0.03** on the August Dataset to **0.23** on the Full Dataset.

### Limitations and the Path Forward

While the results are promising, there are inherent limitations. Most notably, Reddit's user base skews heavily younger, white, and male, meaning it does not perfectly reflect the broader demographics of the United States. Additionally, due to the pay-as-you-go cost of the OpenAI API, the GPT-3 analysis was limited to evaluating only the first ~45 words (60 tokens) of each comment.

Looking ahead, there is massive potential to refine this process. Future research could involve fine-tuning OpenAI models specifically on COVID-19 conversational data or expanding the scope into a time-series analysis to track how public sentiment shifted month by month.

Ultimately, this study confirms that advanced polarity-based sentiment analysis—particularly when applied to individual comments using tools like TextBlob and GPT-3—can serve as a highly effective proxy for real-world vaccine acceptance. For health institutions and policymakers, this means digital chatter can legitimately highlight which regions require targeted education and promotion to protect public health.

**The full paper and repository can be found on my GitHub: [https://github.com/jakecupani/reddit-vaccine](https://github.com/jakecupani/reddit-vaccine)**