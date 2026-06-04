---
layout: default
title: "Retrieval, Ranking & Recommendation"
description: "A hands-on, 24-part series on the full stack of modern recommender systems — from objectives and features to retrieval, ranking, serving, and closing the feedback loop. By Ali Miraftab."
permalink: /retrieval-ranking-recommendation/
---

<section class="hero" style="padding-bottom:1.25rem;">
  <span class="eyebrow fade-in">Series</span>
  <h1 class="fade-in delay-1">Retrieval, Ranking &amp; <span class="grad">Recommendation</span>.</h1>
  <p class="hero__lede fade-in delay-2">
    A hands-on series covering the <strong>full stack</strong> of modern recommender systems —
    from objectives and features to training, retrieval, ranking, serving, and closing the
    feedback loop. Each post is self-contained, with runnable code, math, diagrams, public
    datasets, and production design notes.
  </p>
</section>

<p class="muted" style="font-family:var(--font-mono); font-size:0.9rem; margin-top:0;">
  How to read &nbsp;·&nbsp;
  <strong>Beginners:</strong> 01 → 04 → 06 → 21 &nbsp;·&nbsp;
  <strong>Practitioners:</strong> jump to your problem &nbsp;·&nbsp;
  <strong>System designers:</strong> start with 24, then drill in.
</p>

## The end-to-end mental model

```mermaid
flowchart LR
    A[User Request] --> B[Candidate Generation<br/>Retrieval]
    B --> C[Filtering<br/>Business Rules]
    C --> D[Ranking<br/>Heavy Model]
    D --> E[Re-Ranking<br/>Diversity / MMR]
    E --> F[Serve Top-K]
    F --> G[Logs + Feedback]
    G --> H[Feature Store]
    H --> I[Offline Training]
    I --> B
    I --> D
    G --> J[A/B Test Metrics]
```

## All 24 posts

{% assign series_posts = site.posts | where: "series", "rrr" | sort: "order" -%}
<div class="series-list">{% for post in series_posts %}<a class="series-card" href="{{ post.url | relative_url }}"><span class="num">{% if post.order < 10 %}0{% endif %}{{ post.order }}</span><span class="t">{{ post.title | split: "— " | last }}</span><span class="d">{{ post.theme }} — {{ post.description }}</span></a>{% endfor %}</div>

## Recurring public datasets

<div class="series" markdown="1">

| Dataset | Domain | Scale | Link |
|---|---|---|---|
| MovieLens (100K–25M) | Movies | up to ~25M ratings | [grouplens.org](https://grouplens.org/datasets/movielens/) |
| Amazon Reviews 2018 | E-commerce | ~233M reviews | [nijianmo.github.io](https://nijianmo.github.io/amazon/index.html) |
| Yelp Open Dataset | Local business | ~7M reviews | [yelp.com/dataset](https://www.yelp.com/dataset) |
| H&M Personalized Fashion | Fashion | 31M transactions | [kaggle.com](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) |
| RetailRocket | E-commerce | ~2.7M events | [kaggle.com](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| Criteo 1TB Click Logs | Ads CTR | 4B examples | [ailab.criteo.com](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) |
| Spotify MPD | Playlists | 1M playlists | [aicrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) |
| GoodReads | Books | ~228M interactions | [mengtingwan.github.io](https://mengtingwan.github.io/data/goodreads.html) |
| MIND News | News | high item churn | [msnews.github.io](https://msnews.github.io/) |

</div>
