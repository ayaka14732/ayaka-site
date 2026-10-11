#-*- coding: utf-8 -*-

import html
import json
import os
import re
import subprocess
from pathlib import Path

here = Path(os.path.join(os.path.dirname(os.path.abspath(__file__))))
metadata_file = os.path.join(here, 'pandoc.yaml')
site_url = 'https://ayaka.shn.hk/'
site_name = '綾香的 Blog'
default_image = 'https://avatars2.githubusercontent.com/u/68557794?s=400&u=343da7bebff676129e87341def71121ebffc4c9c&v=4'

def page_url(directory):
	relative = os.path.relpath(directory, here)
	return site_url if relative == '.' else site_url + relative.replace(os.sep, '/') + '/'

def last_modified(src_file):
	# Empty for a page that has not been committed yet
	return subprocess.run(['git', 'log', '-1', '--format=%cI', '--', src_file], cwd=here, capture_output=True, text=True).stdout.strip()

def add_seo_tags(directory, src_file, dst_file):
	'''Add the canonical URL, Open Graph and Twitter Card tags and JSON-LD to a built page. A page can define any of the Open Graph tags itself (in the `math` field of its front matter), which then takes precedence.'''
	with open(dst_file, encoding='utf-8') as f:
		page = f.read()
	head_end = page.index('</head>')
	head = page[:head_end]

	def find(pattern):
		match = re.search(pattern, head)
		return html.unescape(match[1]) if match else None

	def meta(name):
		return find(rf'<meta (?:name|property)="{re.escape(name)}" content="(.*?)" ?/>')

	url = page_url(directory)
	title = find(r'<title>(.*?)</title>')
	description = meta('description')
	image = meta('og:image')
	identifier = meta('dcterms.identifier') # the DOI URL of a paper
	modified = last_modified(src_file)

	properties = {
		'og:title': title,
		'og:type': 'website' if url == site_url else 'article',
		'og:url': url,
		'og:site_name': site_name,
		'og:description': description,
		'og:image': default_image,
	}
	tags = [f'<link rel="canonical" href="{html.escape(url)}" />']
	tags += [f'<meta property="{name}" content="{html.escape(content)}" />' for name, content in properties.items() if content and not meta(name)]
	tags.append(f'<meta name="twitter:card" content="{"summary_large_image" if image else "summary"}" />')

	data = {
		'@context': 'https://schema.org',
		'@type': 'WebSite' if url == site_url else 'ScholarlyArticle' if identifier else 'Article',
		'name' if url == site_url else 'headline': title,
		'url': url,
		'inLanguage': find(r'<html [^>]*?\blang="(.*?)"'),
		'author': {'@type': 'Person', 'name': meta('author'), 'url': site_url},
		'description': description,
		'image': image,
		'datePublished': meta('dcterms.date'),
		'dateModified': modified,
		'sameAs': identifier,
	}
	data = {key: value for key, value in data.items() if value}
	tags.append('<script type="application/ld+json">' + json.dumps(data, ensure_ascii=False).replace('</', '<\\/') + '</script>')

	with open(dst_file, 'w', encoding='utf-8') as f:
		f.write(head + ''.join(f'  {tag}\n' for tag in tags) + page[head_end:])

pages = []

for f in here.glob('./**/index.md'):
	directory = os.path.dirname(f)
	src_file = os.path.join(directory, 'index.md')
	dst_file = os.path.join(directory, 'index.html')
	pages.append((page_url(directory), last_modified(src_file)))

	if not os.path.exists(dst_file) or os.path.getmtime(dst_file) < max(os.path.getmtime(src_file), os.path.getmtime(metadata_file), os.path.getmtime(__file__)):
		print(f'Building {directory}...')
		if os.system(f'pandoc --fail-if-warnings --metadata-file={metadata_file} --listings --toc --toc-depth=2 -s {src_file} -o {dst_file}'):
			print('Build failed. Retrying with --webtex...')
			os.system(f'pandoc --fail-if-warnings --metadata-file={metadata_file} --listings --webtex --toc --toc-depth=2 -s {src_file} -o {dst_file}')
		add_seo_tags(directory, src_file, dst_file)
	else:
		print(f'Skipping {directory}...')

with open(os.path.join(here, 'sitemap.xml'), 'w', encoding='utf-8') as f:
	f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
	for url, modified in sorted(pages):
		f.write(f'  <url><loc>{html.escape(url)}</loc>' + (f'<lastmod>{modified}</lastmod>' if modified else '') + '</url>\n')
	f.write('</urlset>\n')
