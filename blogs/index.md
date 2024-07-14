---
layout: default
title: Blogs
---

# My Blogs

<div class="blog-list">
  <div class="sort-options">
    <label for="sort-select">Sort by:</label>
    <select id="sort-select">
      <option value="date">Date</option>
      <option value="topic">Topic</option>
    </select>
  </div>
  <ul id="blog-list">
    {% for post in site.posts %}
      {{ post.date | date: "%B %d, %Y" }}
      <li data-date="{{ post.date | date_to_xmlschema }}" data-topic="{{ post.topic }}">
        <a href="{{ post.url }}">{{ post.title }}</a>        
      </li>
    {% endfor %}
  </ul>
</div>

