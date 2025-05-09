#!/usr/bin/env python3
"""
Script to copy or generate files from cache to final_articles directory
"""

import os
import glob
import json
import sys
import shutil

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
FINAL_ARTICLES_DIR = os.path.join(BASE_DIR, 'final_articles')
MARKDOWN_DIR = os.path.join(FINAL_ARTICLES_DIR, 'markdown')
HTML_DIR = os.path.join(FINAL_ARTICLES_DIR, 'html')

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

print(f"Cache directory: {CACHE_DIR}")
print(f"Markdown directory: {MARKDOWN_DIR}")
print(f"HTML directory: {HTML_DIR}")

# Find all cached final articles
final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
print(f"Found {len(final_articles)} cached final articles")

# Find markdown files in workspace
tech_dive_md_files = glob.glob(os.path.join(BASE_DIR, 'tech_deep_dive_*.md'))
print(f"Found {len(tech_dive_md_files)} tech_deep_dive markdown files in root directory")

# Find HTML files in workspace
tech_dive_html_files = glob.glob(os.path.join(BASE_DIR, 'tech_deep_dive_*.html'))
print(f"Found {len(tech_dive_html_files)} tech_deep_dive HTML files in root directory")

# Copy tech_deep_dive files to final_articles if they exist
for md_file in tech_dive_md_files:
    dest_file = os.path.join(MARKDOWN_DIR, os.path.basename(md_file))
    print(f"Copying {md_file} to {dest_file}")
    shutil.copy2(md_file, dest_file)

for html_file in tech_dive_html_files:
    dest_file = os.path.join(HTML_DIR, os.path.basename(html_file))
    print(f"Copying {html_file} to {dest_file}")
    shutil.copy2(html_file, dest_file)

# Create markdown and HTML files from cache if they don't exist
for article_path in final_articles:
    try:
        with open(article_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = data.get('content', '')
        if not content:
            print(f"No content in {article_path}")
            continue
            
        # Get title
        title = content.splitlines()[0] if content else 'Unknown Title'
        if title.startswith('# '):
            title = title[2:]  # Remove Markdown heading marker
            
        # Extract the filename and ID
        filename = os.path.basename(article_path)
        article_id = filename.replace('final_article_', '').replace('.json', '')
        
        # Create paths for markdown and HTML files
        markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
        html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{article_id}.html")
        
        # Create markdown file if it doesn't exist
        if not os.path.exists(markdown_path):
            try:
                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(content)
                print(f"Created markdown file: {markdown_path}")
            except Exception as e:
                print(f"Error creating markdown file {markdown_path}: {e}")
                
        # Create HTML file if it doesn't exist
        if not os.path.exists(html_path):
            try:
                # Simple conversion of markdown to HTML
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
            font-size: 1.8em;
        }}
        h3 {{
            color: #16a085;
            font-size: 1.4em;
        }}
        h4 {{
            color: #c0392b;
            font-size: 1.2em;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            background: #f5f5f5;
            border-left: 5px solid #3498db;
            padding: 10px 20px;
            margin: 20px 0;
        }}
        code {{
            background: #eee;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    {content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")}
</body>
</html>"""
                
                with open(html_path, 'w', encoding='utf-8') as html_file:
                    html_file.write(html_content)
                print(f"Created HTML file: {html_path}")
            except Exception as e:
                print(f"Error creating HTML file {html_path}: {e}")
                
    except Exception as e:
        print(f"Error processing cache file {article_path}: {e}")

# Count files in each directory after operation
md_files = glob.glob(os.path.join(MARKDOWN_DIR, '*.md'))
html_files = glob.glob(os.path.join(HTML_DIR, '*.html'))

print(f"\nSummary:")
print(f"- {len(final_articles)} cache files found")
print(f"- {len(md_files)} markdown files in final_articles/markdown")
print(f"- {len(html_files)} HTML files in final_articles/html")

# Create dummy test files if requested
if len(sys.argv) > 1 and sys.argv[1] == '--create-test':
    print("\nCreating test article files...")
    
    # Create a test article
    test_content = """# Test Article: AI in Modern Software Development

## Introduction

This is a test article about artificial intelligence in software development. It covers the basics of using AI tools to enhance productivity.

## Key Concepts

### Machine Learning Integration

Machine learning can be integrated with modern IDEs to provide code suggestions and automated refactoring.

### Natural Language Processing

NLP allows developers to write documentation and even code using natural language prompts.

## Conclusion

AI tools are becoming an essential part of the software development lifecycle, enhancing productivity and code quality.

## Further Exploration

Want to dive deeper into this topic? Here are some thought-provoking questions to explore:

1. How will AI affect programming jobs in the next decade?
2. What ethical considerations should be made when integrating AI in development tools?
3. How can traditional development methodologies adapt to AI-assisted programming?
4. What skills will developers need to effectively work with AI tools?
5. How might programming languages evolve to better support AI integration?

Feel free to research these questions and share your findings!
"""
    
    # Create timestamp ID
    import time
    timestamp = int(time.time())
    article_id = timestamp
    
    # Save to cache
    cache_path = os.path.join(CACHE_DIR, f"final_article_{article_id}.json")
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({
            'content': test_content,
            'timestamp': timestamp
        }, f)
    print(f"Created test cache file: {cache_path}")
    
    # Save markdown version
    markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    print(f"Created test markdown file: {markdown_path}")
    
    # Save HTML version
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Article: AI in Modern Software Development</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
            font-size: 1.8em;
        }}
        h3 {{
            color: #16a085;
            font-size: 1.4em;
        }}
        h4 {{
            color: #c0392b;
            font-size: 1.2em;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            background: #f5f5f5;
            border-left: 5px solid #3498db;
            padding: 10px 20px;
            margin: 20px 0;
        }}
        code {{
            background: #eee;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    {test_content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")}
</body>
</html>"""
    
    html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{article_id}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Created test HTML file: {html_path}")
    
    # Final count after test creation
    md_files = glob.glob(os.path.join(MARKDOWN_DIR, '*.md'))
    html_files = glob.glob(os.path.join(HTML_DIR, '*.html'))
    cache_files = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
    
    print(f"\nFinal counts after test creation:")
    print(f"- {len(cache_files)} cache files")
    print(f"- {len(md_files)} markdown files")
    print(f"- {len(html_files)} HTML files") 