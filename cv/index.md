---
layout: default
title: CV
description: "Ali Miraftab, Ph.D. — Curriculum Vitae. Lead AI/ML Engineer specializing in transformer-based recommendation, RAG ranking, and large-scale information retrieval."
---

<section class="hero" style="padding-bottom:1rem;">
  <span class="eyebrow fade-in">Curriculum Vitae</span>
  <h1 class="fade-in delay-1">Ali Miraftab, Ph.D.</h1>
  <p class="hero__lede fade-in delay-2">
    Lead AI/ML Engineer with 10+ years building production-scale AI systems that drive measurable
    business outcomes. Specialize in transformer-based recommendation, RAG-based ranking,
    personalization, and large-scale information retrieval, with deep experience scaling models
    for consumer products.
  </p>

  <div class="hero__actions fade-in delay-3">
    <a class="btn btn--primary" href="mailto:{{ site.email }}">{{ site.email }}</a>
    <a class="btn" href="https://www.linkedin.com/in/{{ site.linkedin }}" target="_blank" rel="noopener">LinkedIn</a>
    <a class="btn" href="https://github.com/{{ site.github }}" target="_blank" rel="noopener">GitHub</a>
    <a class="btn" href="{{ site.scholar }}" target="_blank" rel="noopener">Google Scholar</a>
  </div>

  <div class="meta-strip fade-in delay-4">
    <span>Austin, TX</span>
    <span>+1 (210) 548-1604</span>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Professional summary</h2>
  </div>
  <p>
    Architected Procore's RAG-and-contextual-intelligence recommendation stack and end-to-end LLM Product Insights Platform; led Dell's semantic supply-chain retrieval / re-rank stack with LLM-based entity resolution; pioneered Expedia's first flight ranking and recommendation systems. Operate as a cross-functional technical leader — owning scope and decision authority end to end, partnering directly with executive, product, business, and engineering leadership, hiring and mentoring teams, and standardizing the AI/ML dev/deployment lifecycle to compress time-to-production. Comfortable in high-ambiguity zero-to-one settings; strong on distributed systems, data infrastructure, real-time / streaming pipelines, and bridging model development with online serving.
  </p>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Leadership &amp; scope</h2>
  </div>
  <div class="grid grid--2">
    <div class="card"><div class="card__kicker">Cross-functional leadership</div><p class="card__body">Partner directly with executive, product, business, and tech leaders to define AI/ML strategy, scope deliverables, set roadmaps, align milestones across orgs, and translate ambiguous business needs into concrete technical solutions.</p></div>
    <div class="card"><div class="card__kicker">People leadership</div><p class="card__body">Hire, mentor, and lead teams of AI/ML engineers and scientists at C5i, Quotograph, Expedia, and Procore; set technical standards, unblock ICs, and own design and code reviews.</p></div>
    <div class="card"><div class="card__kicker">Decision authority &amp; scope ownership</div><p class="card__body">Own end-to-end model and platform decisions from problem framing → data and feature design → modeling and evaluation → serving and infra → production rollout → post-launch iteration.</p></div>
    <div class="card"><div class="card__kicker">Standardized AI/ML lifecycle</div><p class="card__body">Built a repeatable AI/ML guideline covering scoping, data acquisition, EDA, modeling, evaluation, deployment, A/B testing, and monitoring — accelerating delivery cadence from annual to quarterly.</p></div>
    <div class="card"><div class="card__kicker">Roadmap &amp; milestone definition</div><p class="card__body">Set and defended quarterly and annual AI/ML roadmaps; defined milestones, success metrics, and exit criteria; balanced research-grade explorations with delivery commitments.</p></div>
    <div class="card"><div class="card__kicker">Hiring &amp; org-building</div><p class="card__body">Designed hiring rubrics, interview loops, and leveling guidance for AI/ML engineers and scientists.</p></div>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Core expertise</h2>
    <span class="section__sub">// the toolkit</span>
  </div>

  <div class="skills">
    <div class="skill">
      <div class="skill__head">Ranking &amp; personalization</div>
      <ul class="tags">
        <li class="tag">Transformer rec/ranking</li>
        <li class="tag">HSTU</li><li class="tag">SASRec</li><li class="tag">BERT4Rec</li><li class="tag">TransAct</li>
        <li class="tag">DIN / DIEN / SIM</li><li class="tag">BST</li>
        <li class="tag">RAG-based recommendation</li>
        <li class="tag">Generative retrieval · TIGER · semantic IDs</li>
        <li class="tag">Two-Tower deep retrieval-ranking</li>
        <li class="tag">PyTorch · TensorFlow</li>
        <li class="tag">Collaborative filtering · ALS · CF</li>
        <li class="tag">GraphSAGE · GNN</li>
        <li class="tag">LambdaRank · LightGBM · XGBoost</li>
        <li class="tag">MMoE · PLE</li>
        <li class="tag">Contextual bandits / MAB</li>
        <li class="tag">CTR / CVR</li>
        <li class="tag">Position-bias · counterfactual eval</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">Search &amp; retrieval</div>
      <ul class="tags">
        <li class="tag">SBERT bi-encoder + cross-encoder re-rank</li>
        <li class="tag">Multimodal · CLIP</li>
        <li class="tag">Hybrid retrieval</li>
        <li class="tag">HNSW · FAISS · ScaNN · Annoy</li>
        <li class="tag">Databricks Vector Search</li>
        <li class="tag">pgvector · Milvus · Pinecone · Weaviate · Chroma</li>
        <li class="tag">Elasticsearch / OpenSearch</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">LLMs &amp; generative AI</div>
      <ul class="tags">
        <li class="tag tag--solid">RAG + contextual-intelligence</li>
        <li class="tag">Generative recommendation</li>
        <li class="tag">Llama · GPT fine-tuning &amp; serving</li>
        <li class="tag">LoRA · QLoRA · PEFT</li>
        <li class="tag">Instruction tuning</li>
        <li class="tag">ReAct</li>
        <li class="tag">Topic modeling &amp; taxonomy extraction with LLMs</li>
        <li class="tag">Prompt engineering</li>
        <li class="tag">Evaluation harnesses</li>
        <li class="tag">Quantization · kernel fusion · dynamic batching</li>
        <li class="tag">Diffusion-model basics</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">ML infrastructure</div>
      <ul class="tags">
        <li class="tag">Distributed training &amp; feature pipelines</li>
        <li class="tag">Streaming / real-time inference</li>
        <li class="tag">Low-latency online services</li>
        <li class="tag">A/B testing</li>
        <li class="tag">MLflow · Airflow</li>
        <li class="tag">GPU clusters (GCP, AWS, on-prem)</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">Classical ML &amp; forecasting</div>
      <ul class="tags">
        <li class="tag">XGBoost · LightGBM</li>
        <li class="tag">Clustering</li>
        <li class="tag">ARIMA · Prophet · deep forecasters</li>
        <li class="tag">Structured prediction</li>
        <li class="tag">Large-scale optimization</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">Computer vision &amp; NLP</div>
      <ul class="tags">
        <li class="tag">ResNet · VGG</li>
        <li class="tag">YOLO</li>
        <li class="tag">EAST text detection</li>
        <li class="tag">OpenCV</li>
        <li class="tag">BERT-family transformers</li>
        <li class="tag">Word2Vec · GloVe</li>
        <li class="tag">spaCy · NLTK · HuggingFace</li>
      </ul>
    </div>

    <div class="skill">
      <div class="skill__head">Languages &amp; tools</div>
      <ul class="tags">
        <li class="tag">Python</li><li class="tag">PySpark</li><li class="tag">SQL</li><li class="tag">R</li><li class="tag">MATLAB</li>
        <li class="tag">PyTorch · TensorFlow · JAX · Trax</li>
        <li class="tag">HuggingFace · scikit-learn · LangChain</li>
        <li class="tag">Docker</li>
        <li class="tag">Scala · C++ · Java (familiar)</li>
      </ul>
    </div>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Professional experience</h2>
  </div>

  <div class="timeline">

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Lead Machine Learning Engineer</span> · <span class="tl-org">Procore Technologies</span></div>
        <span class="tl-meta">Sep 2024 – Present · Remote</span>
      </div>
      <ul class="tl-list">
        <li>Developed retrieval and ranking systems based on <strong>generative AI and transformer-based recommendation models</strong>; designed and implemented a <strong>RAG architecture with an additional contextual-intelligence layer</strong> to improve product recommendations, ranking quality, and relevance.</li>
        <li>Built parallel <strong>contextual recommendation architectures</strong> generating sales recommendations and talking points for Sales Reps and Customer Success Engineers, grounded in account history, product usage, and support signals.</li>
        <li>For the contextual-intelligence layer, applied <strong>topic modeling, taxonomy extraction, and semantic modeling with LLMs</strong> to enrich understanding, improve retrieval precision, and lift recommendation quality and downstream conversion.</li>
        <li>Architected and shipped an <strong>end-to-end LLM-powered Product Insights Platform</strong> that turns unstructured and structured customer signals (call notes, tickets, reviews, telemetry) into queryable, source-attributed findings.</li>
        <li>Pipeline: unstructured + structured sources → chunking → <strong>Vector Search</strong> embeddings → structured extraction with <strong>Llama 3.3 70B</strong> + JSON schema → queryable Delta table.</li>
        <li>Designed a domain-specific <strong>taxonomy mapping system</strong> (62 tools · 5 product solutions · 36 workflows · 11 outcome categories) with alias resolution and few-shot prompting — <strong>&gt;85% tagging accuracy</strong>. Every finding is traced to a verbatim source quote for full auditability.</li>
        <li>Built a read-only <strong>MCP (Model Context Protocol) server</strong> exposing the insights table via SQL and natural-language-to-SQL translation — enabling ad-hoc analysis from Cursor, Claude, and Slack. Cut strategic-question turnaround from <strong>~1 week of manual synthesis to minutes of querying</strong>.</li>
        <li>Implemented validation: confidence scoring, duplicate detection, source traceability, version control for extraction runs, and incremental processing.</li>
        <li>Deploy, fine-tune, and optimize production LLMs for sales and customer-success automation; apply quantization, kernel fusion, and dynamic batching to reduce latency and GPU spend.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Lead AI/ML, Sr. Manager</span> · <span class="tl-org">C5i (Course5 Intelligence)</span></div>
        <span class="tl-meta">Feb 2024 – Sep 2024</span>
      </div>
      <ul class="tl-list">
        <li>Led the end-to-end design of a <strong>personalized product recommendation and semantic-search platform for Dell.com</strong> and other sales channels; consumer-scale ML over structured and unstructured supply-chain data.</li>
        <li>Architected a two-stage retrieval-and-rank stack: <strong>Sentence-BERT bi-encoder</strong> for candidate generation and a <strong>cross-encoder re-ranker</strong> for relevance, with Word2Vec / GloVe and graph embeddings (GraphSAGE, GNN) over the part / SKU / product hierarchy.</li>
        <li>Designed an <strong>LLM-based entity-resolution pipeline</strong> for part / SKU / product deduplication across heterogeneous supply-chain records:
          <ul>
            <li>Retrieval: SBERT + HNSW / FAISS ANN search (O(log n) candidates at <strong>~95% recall</strong>).</li>
            <li>Ranking: gradient-boosted classifier over multi-feature similarity (cosine, Jaccard, token overlap, edit distance, length ratio, entity-type match).</li>
            <li>Clustering: hierarchical agglomerative with complete linkage and confidence scoring; canonical-name selection per cluster.</li>
            <li>Production: incremental batching, streaming-friendly DBSCAN fallback, and a typed Python <code>EntityResolver</code> API.</li>
          </ul>
        </li>
        <li>Fine-tuned Llama and GPT models with <strong>LoRA / QLoRA / PEFT</strong>; built offline evaluation harnesses and online experimentation patterns; defined serving blueprints for downstream teams.</li>
        <li>Owned scope, roadmap, and milestone definitions for the Dell engagement; led ML engineers and scientists; partnered with Dell leadership on requirements and rollout.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Lead AI/ML</span> · <span class="tl-org">Quotograph</span></div>
        <span class="tl-meta">Oct 2023 – Feb 2024</span>
      </div>
      <ul class="tl-list">
        <li>Built a <strong>multimodal semantic-search system on CLIP</strong> for automated quote / image generation; joint image + text embeddings powering personalized, conversational discovery.</li>
        <li>Applied instruction fine-tuning, LoRA / QLoRA PEFT, and structured prompt engineering for domain adaptation; designed an evaluation harness aligned with qualitative product goals.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Staff → Senior → Machine Learning Scientist</span> · <span class="tl-org">Expedia Group</span></div>
        <span class="tl-meta">Sep 2017 – Oct 2023 · Austin, TX</span>
      </div>
      <div class="tl-sub">Staff MLS — Ranking, Recommendation, NLP (Apr 2019 – Oct 2023); Senior MLS — Ranking (Jan – Mar 2019); MLS — Computer Vision (Sep 2017 – Dec 2018).</div>

      <h4 class="mt-2" style="font-size:0.95rem; color:var(--text); font-family:var(--font-display);">Flights — Ranking, Recommendation, Personalization</h4>
      <ul class="tl-list">
        <li>Designed and shipped Expedia's <strong>first personalized flight recommendation model on iOS SERP</strong>, connecting users to the most relevant flights under high-throughput / low-latency online serving constraints. <strong>+0.7% conversion rate</strong>. Algorithms: LambdaRank (LightGBM) and Two-Tower retrieval-ranking deep model (TensorFlow).</li>
        <li>Built the <strong>Web Flights LambdaRank LTR</strong> ranker to lift revenue. <strong>+1.2% GP/order</strong>.</li>
        <li>Predictive cache-fill DNN (TensorFlow) — <strong>2× cache hit-rate</strong>, reducing downstream pricing-call volume and infra cost.</li>
        <li>CTR predictor gating flight pricing calls to reduce downstream traffic.</li>
        <li>Price forecasting and alerts using XGBoost — recommend the best time to book.</li>
        <li>Hero-image selection driven by a contextual bandit for SERP relevance.</li>
      </ul>

      <h4 class="mt-2" style="font-size:0.95rem; color:var(--text); font-family:var(--font-display);">Vacation Rentals &amp; Lodging — Ranking and CV</h4>
      <ul class="tl-list">
        <li>Neural ranker in PyTorch for Vacation Rentals SERP.</li>
        <li>Urgency-messaging XGBoost model predicting property / destination occupancy for personalization.</li>
        <li>Computer-vision stack feeding ranking and marketplace health: image aesthetic scoring (ResNet-152 / VGG-16), YOLO on-demand amenity detection, PyTorch scene classification (ResNet-152), fraud detection combining NLP + EAST text detector. Outputs powered image ordering, hero-image MAB, LTR features, marketplace-health analytics, and CVR boosting in email and ads.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Machine Learning &amp; Cloud Researcher</span> · <span class="tl-org">UTSA — Open Cloud Institute</span></div>
        <span class="tl-meta">Jan 2015 – Sep 2017</span>
      </div>
      <ul class="tl-list">
        <li>Fusing IoT contents with geosocial networks for anomalous-behavior detection in smart communities — won a <strong>$100K Cisco grant</strong>.</li>
        <li>Designed intelligent <strong>workload-aware schedulers for multi-cloud environments</strong> — resulted in <strong>U.S. Patent 10,452,451 B2</strong>.</li>
        <li>Published a comprehensive open-source TensorFlow tutorial covering computer vision and NLP for the UTSA ML6973 course.</li>
        <li>Side projects: EEG emotion classification, facial-image emotion detection, sentiment analysis on movie reviews, music classification with CNN + RNN, vehicle make/model recognition, marker detection.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">RAN Conceptual Planner Engineer</span> · <span class="tl-org">MCI / Hamrah-e Aval</span></div>
        <span class="tl-meta">Jun 2012 – Aug 2014 · Tehran, Iran</span>
      </div>
      <ul class="tl-list">
        <li>Designed WCDMA and LTE RAN systems in collaboration with Ericsson and Huawei; GSM / GPRS / EDGE network expansion and NSN license balancing.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Communication Engineer (Consultant)</span> · <span class="tl-org">Darya Pala Energy</span></div>
        <span class="tl-meta">Jan 2011 – May 2012</span>
      </div>
      <ul class="tl-list">
        <li>Designed end-to-end telecommunication systems (fiber, wireless, LAN, CCTV) for oil and gas sites.</li>
      </ul>
    </div>

    <div class="tl-item">
      <div class="tl-head">
        <div><span class="tl-role">Microwave &amp; Electronic Engineer</span> · <span class="tl-org">TTQC — Healthcare</span></div>
        <span class="tl-meta">Dec 2010 – Jun 2012</span>
      </div>
      <ul class="tl-list">
        <li>Collaborated with HITEC Poland on radiation-therapy solutions for cancer treatment.</li>
      </ul>
    </div>

  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Education</h2>
  </div>
  <div class="grid grid--3">
    <div class="card">
      <div class="card__kicker">2014 – 2017</div>
      <h3 class="card__title">Ph.D., ECE — AI &amp; Big Data</h3>
      <p class="card__body">The University of Texas at San Antonio. Dissertation: <em>Real-Time Adaptive Data-Driven Perception for Anomaly Priority Scoring at Scale.</em></p>
    </div>
    <div class="card">
      <div class="card__kicker">2008 – 2010</div>
      <h3 class="card__title">M.Sc., EE — Microwave &amp; Optical Communication</h3>
      <p class="card__body">Sharif University of Technology, Tehran.</p>
    </div>
    <div class="card">
      <div class="card__kicker">2002 – 2007</div>
      <h3 class="card__title">B.Sc., EE — Electronics</h3>
      <p class="card__body">Semnan University.</p>
    </div>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Patent</h2>
  </div>
  <div class="card">
    <div class="card__kicker">US 10,452,451 B2 · October 22, 2019</div>
    <h3 class="card__title">Systems and Methods for Scheduling of Workload-Aware Jobs on Multi-Clouds</h3>
    <p class="card__body muted">Miraftabzadeh, S. A., and Najafirad, P.</p>
  </div>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Selected publications</h2>
  </div>
  <ul class="tl-list" style="padding-left:0; list-style:none;">
    <li>Miraftabzadeh, S. A., Rad, P., Choo, K.-K. R., &amp; Jamshidi, M. <strong>A privacy-aware architecture at the edge for autonomous real-time identity re-identification in crowds.</strong> <em>IEEE Internet of Things Journal,</em> 2017.</li>
    <li>Miraftabzadeh, S. A., Rad, P., &amp; Jamshidi, M. <strong>Distributed Algorithm with Inherent Intelligence for Multi-cloud Resource Provisioning.</strong> In <em>Intelligent Decision Support Systems for Sustainable Computing,</em> Springer, 2017, pp. 77–99.</li>
    <li>Miraftabzadeh, S. A., Rad, P., Jamshidi, M., &amp; Prevost, J. <strong>Customer Review Analytics using Subjective Loss Function for Conceptual-based Learning.</strong> 13th SoSE, IEEE, 2018.</li>
    <li>Miraftabzadeh, S. A., Rad, P., Jamshidi, M., &amp; Prevost, J. <strong>The Subjective Loss Function for Conceptual-based Customer Reviewer Analytics.</strong> WAC, 2018.</li>
    <li>Miraftabzadeh, S. A., Rad, P., &amp; Jamshidi, M. <strong>Temporal Face Embedding as Biometric Tokenization for Decentralized IoT.</strong> WAC, 2017.</li>
    <li>Miraftabzadeh, S. A., Rad, P., &amp; Jamshidi, M. <strong>Efficient distributed algorithm for scheduling workload-aware jobs on multi-clouds.</strong> 11th SoSE, IEEE, 2016.</li>
    <li>Miratabzadeh, S. A., Gallardo, N., Gamez, N., Haradi, K., Puthussery, A. R., Rad, P., &amp; Jamshidi, M. <strong>Cloud robotics: A software architecture for heterogeneous large-scale autonomous robots.</strong> WAC, 2016.</li>
  </ul>
</section>

<section class="section">
  <div class="section__head">
    <h2 class="section__title">Honors &amp; awards</h2>
  </div>
  <div class="grid grid--2">
    <div class="card"><div class="card__kicker">Federal fellowship</div><p class="card__body"><strong>NSF Graduate Research Fellowship</strong> — Grant No. 1419165 (supported Ph.D. research).</p></div>
    <div class="card"><div class="card__kicker">Federal grant</div><p class="card__body">Partial Ph.D. support by <strong>Air Force Research Laboratory &amp; OSD</strong> (Grant FA8750-15-2-0116).</p></div>
    <div class="card"><div class="card__kicker">Industry grant</div><p class="card__body"><strong>$100K Cisco grant</strong> — IoT + geosocial anomaly detection in smart communities.</p></div>
    <div class="card"><div class="card__kicker">Scholarship</div><p class="card__body">Open Cloud Institute Outstanding Student Scholarship (2015 – 2017); Lutcher Brown Scholarship and Distinguished Reward, UTSA; ECE Teaching Assistantship, UTSA.</p></div>
  </div>
</section>
