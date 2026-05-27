---
layout: default
title: Writing
description: "Notes on RAG, LLM fine-tuning, recommendation systems, and generative AI — by Ali Miraftab."
---

<section class="hero" style="padding-bottom:1.5rem;">
  <span class="eyebrow fade-in">Writing</span>
  <h1 class="fade-in delay-1">Notes from <span class="grad">the lab</span>.</h1>
  <p class="hero__lede fade-in delay-2">
    Working notes on RAG architecture, LLM fine-tuning, transformer-based recommendation,
    and the practical edges of taking AI systems to production.
  </p>
</section>

<section class="section">
  {% assign topics = site.posts | map: "topic" | uniq | compact %}
  <div class="filters">
    <button class="filter is-active" data-topic="all">All</button>
    {% for t in topics %}
      <button class="filter" data-topic="{{ t }}">{{ t }}</button>
    {% endfor %}
  </div>

  <ul class="post-list" id="post-list">
    {% for post in site.posts %}
      <li>
        <a class="post-card" href="{{ post.url | relative_url }}" data-topic="{{ post.topic }}">
          <div class="post-card__meta">
            <span class="post-card__topic">{{ post.topic | default: "Notes" }}</span>
            <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time>
          </div>
          <h3 class="post-card__title">{{ post.title }}</h3>
        </a>
      </li>
    {% endfor %}
  </ul>
</section>
