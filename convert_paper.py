import markdown
import sys
import re

try:
    with open('paper.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Pre-process math blocks to protect them from markdown parser
    # This is a simple hack; a proper plugin is better but this works for simple cases
    # We replace $$...$$ with a placeholder
    math_blocks = []
    def replace_math(match):
        math_blocks.append(match.group(0))
        return f"__MATH_BLOCK_{len(math_blocks)-1}__"
    
    # content = re.sub(r'\$\$(.*?)\$\$', replace_math, content, flags=re.DOTALL)

    html = markdown.markdown(content, extensions=['tables', 'fenced_code'])

    # Restore math blocks and wrap in MathJax delimiters
    # for i, block in enumerate(math_blocks):
    #     # Convert $$...$$ to \[...\] for MathJax
    #     block = block.replace('$$', '\\[', 1).replace('$$', '\\]', 1)
    #     html = html.replace(f"__MATH_BLOCK_{i}__", block)

    # Simple regex replacement for remaining $$ to \[ \] for display math
    # and $ to \( \) for inline math (be careful with this one)
    # Actually, MathJax configuration usually handles $$ by default if configured.
    
    html_template = f'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Paper Preview</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }}
h1, h2, h3 {{ color: #2c3e50; margin-top: 1.5em; }}
h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
th {{ background-color: #f8f9fa; font-weight: 600; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
code {{ background-color: #f1f1f1; padding: 2px 4px; border-radius: 4px; font-family: Consolas, monospace; }}
pre {{ background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }}
img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
blockquote {{ border-left: 4px solid #dfe2e5; margin: 0; padding-left: 16px; color: #6a737d; }}
</style>
</head>
<body>
{html}
</body>
</html>
'''
    with open('paper.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    print("Successfully generated paper.html")

except Exception as e:
    print(f"Error: {e}")

