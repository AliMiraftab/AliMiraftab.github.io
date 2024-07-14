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
  <ul>
    {% for post in site.posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
      <span>{{ post.date | date: "%B %d, %Y" }}</span>
    {% endfor %}
  </ul>
  <!-- <ul id="blog-list">
    <li data-date="2024-07-01" data-topic="Tech"><a href="/2024-07-13-my-first-post/blog1">Blog Title 1</a></li>
    <li data-date="2024-06-25" data-topic="Science"><a href="/blogs/blog2">Blog Title 2</a></li>
    <li data-date="2024-07-10" data-topic="Tech"><a href="/blogs/blog3">Blog Title 3</a></li>
  </ul> -->
</div>

