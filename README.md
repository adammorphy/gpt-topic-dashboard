# gpt-topic-dashboard

This repository contains a python Dash application using GPT 3.5 (text-davinci-003 from OpenAI) for topic and sentiment modeling intended to be used on employee survey responses.

1. Input a valid OpenAI API key
2. Upload a dataset (.csv) which contains 1 column called `understanding_comments`, where each row represents 1 string of text.
3. Wait for GPT to generate topics and review results in the given table and plots.

Topics are limited to Culture, Nature of Job, Manager, Leadership, Compensation, Stress, Burnout, Location, Safety, Inclusion, Relationships, Career Advancement, Learning and Development, Work Life Balence, Satisfaction, or Other. Sentiments are limited to Positive, Mixed, or Negative.
