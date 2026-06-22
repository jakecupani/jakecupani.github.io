---
title: Using Reddit Comment Sentiment Analysis as a Proxy for COVID-19 Vaccine Acceptance by State  ☄️
publishDate: August 1st, 2022
description: 
tags: ['Data Science 📈']
---

# Decoding Digital Chatter: Can Reddit Sentiment Predict COVID-19 Vaccine Acceptance?

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

### References

1. Safety of covid-19 vaccines. (n.d.). Retrieved March 17, 2022, from www.cdc.gov/coronavirus/2019-ncov/vaccines/safety/safety-of-vaccines.html
2. Covid-19 vaccine confidence. (2022, February 28). Retrieved March 17, 2022, from [https://www.cdc.gov/vaccines/covid-19/vaccinate-with-confidence.html](https://www.cdc.gov/vaccines/covid-19/vaccinate-with-confidence.html)
3. Guille, A., Hacid, H., Favre, C., Zighed, D. A. (2013). Information diffusion in online social networks. ACM SIGMOD Record, 42(2), 17-28. doi:10.1145/2503792.2503797
4. Loomba, S., De Figueiredo, A., Piatek, S. J., De Graaf, K., & Larson, H. J. (2021). Measuring the impact of COVID-19 vaccine misinformation on vaccination intent in the UK and USA. Nature Human Behaviour, 5(3), 337-348. doi:10.1038/s41562-021-01056-1
5. Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on reddit during COVID-19: Observational study. Journal of Medical Internet Research, 22(10). doi:10.2196/22635
6. Melton, C. A., Olusanya, O. A., Ammar, N., Shaban-Nejad, A. (2021). Public sentiment analysis and topic modeling regarding COVID-19 vaccines on the Reddit social media platform: A call to action for strengthening vaccine confidence. Journal of Infection and Public Health, 14(10), 1505-1512. doi:10.1016/j.jiph.2021.08.010
7. Yan, C., Law, M., Nguyen, S., Cheung, J., & Kong, J. (2021). Comparing public sentiment toward covid-19 vaccines across Canadian cities: Analysis of comments on Reddit. Journal of Medical Internet Research, 23(9). doi:10.2196/32685
8. COVID-19 Vaccinations in the United States. (2022, March 17). Retrieved March 17, 2022, from [https://covid.cdc.gov/covid-data-tracker/#vaccinations_vacc-people-onedose-pop-5yr](https://www.google.com/search?q=https://covid.cdc.gov/covid-data-tracker/%23vaccinations_vacc-people-onedose-pop-5yr)
9. Report of the sage - World Health Organization. (2014, October 1). Retrieved March 18, 2022
10. Alamoodi, A. H., Zaidan, B. B., Zaidan, A. A., Albahri, O. S., Mohammed, K. I., Malik, R. Q., ... Alaa, M. (2021). Sentiment analysis and its applications in fighting COVID-19 and infectious diseases: A systematic review. Expert systems with applications, 167, 114155.
11. Murray, C., Mitchell, L., Tuke, J., Mackay, M. (2020). Symptom extraction from the narratives of personal experiences with COVID-19 on Reddit. arXiv preprint arXiv:2005.10454.
12. Yin, H., Song, X., Yang, S., Li, J. (2022). Sentiment analysis and topic modeling for COVID-19 vaccine discussions. World Wide Web, 1-17.
13. Kim, J., Hastak, M. (2018). Social network analysis: Characteristics of online social networks after a disaster. International journal of information management, 38(1), 86-96.
14. Alamoodi, A. H., Zaidan, B. B., Al-Masawa, M., Taresh, S. M., Noman, S., Ahmaro, I. Y.,... Salahaldin, A. (2021). Multi-perspectives systematic review on the applications of sentiment analysis for vaccine hesitancy. Computers in Biology and Medicine, 139, 104957.
15. Covid Data Tracker. (2022, April 14). Retrieved April 14, 2022, from [https://covid.cdc.gov/covid-data-tracker/#datatracker-home](https://www.google.com/search?q=https://covid.cdc.gov/covid-data-tracker/%23datatracker-home)
16. COVID-19 Vaccinations in the United States Jurisdiction. (2022, April 14), from [https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-Jurisdi/unsk-b7fc](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-Jurisdi/unsk-b7fc)
17. TextBlob: Simplified Text Processing. (2022, April 14), from [https://textblob.readthedocs.io/en/dev/index.html](https://textblob.readthedocs.io/en/dev/index.html)
18. Chipidza W. (2021). The effect of toxicity on COVID-19 news network formation in political subcommunities on Reddit: An affiliation network approach. International journal of information management, 61, 102397. [https://doi.org/10.1016/j.ijinfomgt.2021.102397](https://doi.org/10.1016/j.ijinfomgt.2021.102397)
19. Open Al Fine Tuning. (2022, May 10). Retrieved May 10, 2022, from [https://beta.openai.com/docs/guides/fine-tuning](https://beta.openai.com/docs/guides/fine-tuning)
20. Yue, L., Chen, W., Li, X., Zuo, W., Yin, M. (2019). A survey of sentiment analysis in social media. Knowledge and Information Systems, 60(2), 617-663.
21. What is GPT-3? (2021, June), from [www.techtarget.com/searchenterpriseai/definition/GPT-3](https://www.techtarget.com/searchenterpriseai/definition/GPT-3)

**The full paper and repository can be found on my GitHub: [https://github.com/jakecupani/reddit-vaccine](https://github.com/jakecupani/reddit-vaccine)**