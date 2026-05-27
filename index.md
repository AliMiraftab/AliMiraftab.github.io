---
layout: default
title: Home
description: "Lead AI/ML Engineer building production-scale recommendation, RAG, and large-scale information-retrieval systems."
---

<section class="hero">
  <span class="eyebrow fade-in">Available for senior AI/ML roles</span>
  <h1 class="fade-in delay-1">
    Hi, I'm Ali — I build <span class="grad">production-scale AI</span> that moves the metrics that matter.
  </h1>
  <p class="hero__lede fade-in delay-2">
    Lead AI/ML Engineer with 10+ years architecting transformer-based recommendation, RAG ranking,
    personalization, and large-scale information retrieval. Currently at
    <strong>Procore</strong>, where I shipped the RAG + contextual-intelligence recommendation
    stack and an end-to-end LLM Product Insights Platform. Previously led ranking at
    <strong>Expedia</strong> and semantic supply-chain retrieval for <strong>Dell</strong>.
  </p>

  <div class="hero__actions fade-in delay-3">
    <a class="btn btn--primary" href="{{ '/cv/' | relative_url }}">
      View CV
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
    </a>
    <a class="btn" href="mailto:{{ site.email }}">Get in touch</a>
    <a class="btn" href="https://www.linkedin.com/in/{{ site.linkedin }}" target="_blank" rel="noopener">LinkedIn</a>
    <a class="btn" href="https://github.com/{{ site.github }}" target="_blank" rel="noopener">GitHub</a>
  </div>

  <div class="meta-strip fade-in delay-4">
    <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg> Austin, TX</span>
    <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Open to remote</span>
    <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/></svg> Ph.D., AI &amp; Big Data — UTSA</span>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Impact in numbers</h2>
    <span class="section__sub">// shipped, measured, in production</span>
  </div>
  <div class="grid grid--3">
    <div class="card stat">
      <div class="stat__value">+1.2%</div>
      <div class="stat__label">GP / order lift from Web Flights LambdaRank LTR at Expedia.</div>
    </div>
    <div class="card stat">
      <div class="stat__value">+0.7%</div>
      <div class="stat__label">Conversion lift from Expedia's first personalized flight recommender on iOS.</div>
    </div>
    <div class="card stat">
      <div class="stat__value">2×</div>
      <div class="stat__label">Cache hit-rate from a predictive cache-fill DNN — cut downstream pricing-call infra cost.</div>
    </div>
    <div class="card stat">
      <div class="stat__value">&gt;85%</div>
      <div class="stat__label">Tagging accuracy of the domain-specific LLM taxonomy mapper (62 tools · 11 outcomes) at Procore.</div>
    </div>
    <div class="card stat">
      <div class="stat__value">~95%</div>
      <div class="stat__label">Recall on SBERT + HNSW/FAISS candidate generation for Dell's entity-resolution pipeline.</div>
    </div>
    <div class="card stat">
      <div class="stat__value">1 week → minutes</div>
      <div class="stat__label">Strategic-question turnaround after shipping the LLM Product Insights MCP server.</div>
    </div>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">What I'm working on</h2>
    <span class="section__sub">// current focus</span>
  </div>
  <div class="grid grid--2">
    <div class="card">
      <div class="card__kicker">Procore · Now</div>
      <h3 class="card__title">RAG + contextual-intelligence recommendations</h3>
      <p class="card__body">Retrieval and ranking on transformer recommenders with an additional contextual-intelligence layer — topic modeling, taxonomy extraction, semantic modeling with LLMs — to lift recommendation quality and downstream conversion for Sales and Customer Success teams.</p>
    </div>
    <div class="card">
      <div class="card__kicker">Procore · Now</div>
      <h3 class="card__title">LLM Product Insights Platform</h3>
      <p class="card__body">End-to-end pipeline turning unstructured + structured customer signals (call notes, tickets, reviews, telemetry) into queryable, source-attributed findings. Llama 3.3 70B structured extraction · Vector Search · a read-only MCP server queryable from Cursor, Claude, and Slack.</p>
    </div>
    <div class="card">
      <div class="card__kicker">Themes</div>
      <h3 class="card__title">Generative retrieval &amp; transformer ranking</h3>
      <p class="card__body">HSTU, SASRec, BERT4Rec, TransAct, DIN/DIEN/SIM, BST; generative retrieval with TIGER and semantic IDs; Two-Tower deep retrieval-ranking; multi-task heads (MMoE, PLE); position-bias and counterfactual evaluation.</p>
    </div>
    <div class="card">
      <div class="card__kicker">Themes</div>
      <h3 class="card__title">Serving at scale</h3>
      <p class="card__body">Bridging model development with online serving — distributed training, streaming / real-time inference, low-latency online services, A/B testing, MLflow, Airflow, GPU clusters on GCP / AWS and on-prem.</p>
    </div>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Selected experience</h2>
    <span class="section__sub">// 10+ years</span>
  </div>
  <div class="grid grid--3">
    <div class="card">
      <div class="card__kicker">Sep 2024 — Present</div>
      <h3 class="card__title">Procore Technologies</h3>
      <p class="card__body">Lead Machine Learning Engineer. RAG + contextual-intelligence recommendations; LLM Product Insights Platform; MCP server for NL-to-SQL analytics.</p>
    </div>
    <div class="card">
      <div class="card__kicker">Sep 2017 — Oct 2023</div>
      <h3 class="card__title">Expedia Group</h3>
      <p class="card__body">Staff → Senior → ML Scientist. Shipped Expedia's first personalized flight recommender, Web Flights LTR, predictive cache-fill, CV stack for Vacation Rentals.</p>
    </div>
    <div class="card">
      <div class="card__kicker">Feb 2024 — Sep 2024</div>
      <h3 class="card__title">C5i · for Dell</h3>
      <p class="card__body">Lead AI/ML, Sr. Manager. Two-stage SBERT bi-encoder + cross-encoder re-rank stack and an LLM-based entity-resolution pipeline for Dell.com supply-chain search.</p>
    </div>
  </div>
  <p class="muted mt-3" style="font-size:0.92rem;">
    More on the <a href="{{ '/cv/' | relative_url }}">CV page</a> — including Quotograph, UTSA Open Cloud Institute, the U.S. Patent on workload-aware multi-cloud scheduling, and selected IEEE / Springer publications.
  </p>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Recent writing</h2>
    <span class="section__sub">// notes from the lab</span>
  </div>
  <ul class="post-list">
    {% for post in site.posts limit:4 %}
    <li>
      <a class="post-card" href="{{ post.url | relative_url }}" data-topic="{{ post.topic }}">
        <div class="post-card__meta">
          <span class="post-card__topic">{{ post.topic | default: "Notes" }}</span>
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %d, %Y" }}</time>
        </div>
        <h3 class="post-card__title">{{ post.title }}</h3>
      </a>
    </li>
    {% endfor %}
  </ul>
  <p class="mt-3"><a href="{{ '/blogs/' | relative_url }}">All posts →</a></p>
</section>
