---
layout: default
title: My Project Blog
---

<ul class="post-list">
  {% for post in site.posts %}
  <li>
    <span class="post-date">{{ post.date | date: "%B %-d, %Y" }}</span>
    <span class="post-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></span>
    {% if post.excerpt %}
    <p class="post-excerpt">{{ post.excerpt | strip_html | truncatewords: 28 }}</p>
    {% endif %}
  </li>
  {% endfor %}
</ul>
