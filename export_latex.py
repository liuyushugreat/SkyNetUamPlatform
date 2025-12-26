import re
import os
import zipfile

def convert_md_to_latex(md_content):
    # 1. Basic Setup
    latex_content = r"""\documentclass[twoside,10pt]{ctexart}
\usepackage[a4paper, left=2cm, right=2cm, top=2cm, bottom=2cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{titlesec}
\usepackage{abstract}

% Title and Author
\title{\heiti 面向城市复杂空域的改进MADDPG多无人机协同避障与轨迹规划}
\author{\kaishu 作者姓名 \\ (单位名称)}
\date{\today}

\begin{document}
\maketitle

"""
    
    lines = md_content.split('\n')
    
    in_abstract = False
    in_ref = False
    in_table = False
    table_lines = []
    
    # Extract Abstract and Keywords
    abstract_zh = ""
    keywords_zh = ""
    abstract_en = ""
    keywords_en = ""
    
    # Simple state machine for parsing
    body_content = ""
    
    iterator = iter(lines)
    for line in iterator:
        line = line.strip()
        
        # Skip title in body (already handled)
        if line.startswith("# "):
            continue
            
        # Abstract extraction
        if "**摘要**：" in line:
            abstract_zh = line.replace("**摘要**：", "")
            continue
        if "**关键词**：" in line:
            keywords_zh = line.replace("**关键词**：", "")
            continue
        if "**Abstract**:" in line:
            abstract_en = line.replace("**Abstract**:", "")
            continue
        if "**Keywords**:" in line:
            keywords_en = line.replace("**Keywords**:", "")
            
            # Flush abstract section to latex
            latex_content += r"\begin{abstract}" + "\n"
            latex_content += abstract_zh + "\n\n"
            latex_content += r"\textbf{关键词：}" + keywords_zh + "\n"
            latex_content += r"\end{abstract}" + "\n\n"
            
            latex_content += r"\renewcommand{\abstractname}{Abstract}" + "\n"
            latex_content += r"\begin{abstract}" + "\n"
            latex_content += abstract_en + "\n\n"
            latex_content += r"\textbf{Keywords: }" + keywords_en + "\n"
            latex_content += r"\end{abstract}" + "\n\n"
            continue
            
        if line == "---":
            continue

        # Section Headers
        if line.startswith("## "):
            if "参考文献" in line:
                in_ref = True
                body_content += r"\begin{thebibliography}{99}" + "\n"
                continue
            header = line.replace("## ", "")
            body_content += f"\\section{{{header}}}\n"
            continue
            
        if line.startswith("### "):
            header = line.replace("### ", "")
            body_content += f"\\subsection{{{header}}}\n"
            continue
            
        if line.startswith("#### "):
            header = line.replace("#### ", "")
            body_content += f"\\subsubsection{{{header}}}\n"
            continue
            
        # Images: ![Caption](filename)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if img_match:
            caption = img_match.group(1)
            filename = img_match.group(2)
            body_content += "\\begin{figure}[H]\n"
            body_content += "\\centering\n"
            body_content += f"\\includegraphics[width=0.8\\textwidth]{{{filename}}}\n"
            body_content += f"\\caption{{{caption}}}\n"
            body_content += "\\end{figure}\n"
            continue
            
        # Math: $$...$$ -> \begin{equation}...\end{equation}
        # Note: simplistic replacement, assuming multiline math is $$ on separate lines
        if line == "$$":
            body_content += "\\begin{equation}\n"
            continue
        
        # Tables
        if line.startswith("|"):
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            continue
        elif in_table:
            # End of table, process it
            in_table = False
            body_content += process_table(table_lines)
            table_lines = []
            
        # References content
        if in_ref:
            if line.startswith("["):
                # Extract key and content
                # [1] LOWE ... -> \bibitem{ref1} LOWE ...
                ref_match = re.match(r'\[(\d+)\]\s*(.*)', line)
                if ref_match:
                    ref_id = ref_match.group(1)
                    ref_text = ref_match.group(2)
                    body_content += f"\\bibitem{{ref{ref_id}}} {ref_text}\n"
            continue

        # Bold text
        line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
        
        # Inline math
        # Replace single $ with \( \) but be careful not to break other things. 
        # Simple heuristic: if line has $ and not $$, replace
        # This is risky with complex regex, so we'll do a simple split/join
        if '$' in line:
            parts = line.split('$')
            new_line = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    new_line += part
                else:
                    new_line += f"${part}$" # Keep standard latex math delimiters
            line = new_line

        if line.strip() != "":
            body_content += line + "\n\n"

    if in_ref:
        body_content += r"\end{thebibliography}" + "\n"

    latex_content += body_content
    latex_content += r"\end{document}"
    
    return latex_content

def process_table(lines):
    # A very simple markdown table to latex converter
    # Input: list of strings (rows)
    # Output: latex tabular string
    
    if len(lines) < 3: return ""
    
    # Determine columns
    header = lines[0]
    cols = header.count('|') - 1
    
    latex_tbl = "\\begin{table}[H]\n\\centering\n"
    latex_tbl += "\\begin{tabular}{" + "c" * cols + "}\n"
    latex_tbl += "\\toprule\n"
    
    # Header
    header_content = header.strip('|').split('|')
    header_str = " & ".join([h.strip() for h in header_content]) + " \\\\\n"
    latex_tbl += header_str
    latex_tbl += "\\midrule\n"
    
    # Body
    for i in range(2, len(lines)):
        row = lines[i]
        row_content = row.strip('|').split('|')
        row_str = " & ".join([r.strip() for r in row_content]) + " \\\\\n"
        latex_tbl += row_str
        
    latex_tbl += "\\bottomrule\n"
    latex_tbl += "\\end{tabular}\n"
    latex_tbl += "\\caption{数据表}\n" # Placeholder caption
    latex_tbl += "\\end{table}\n\n"
    
    return latex_tbl

def create_zip_package():
    # Read md file
    try:
        with open('paper.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print("Error: paper.md not found")
        return

    # Convert
    tex_content = convert_md_to_latex(md_content)
    
    # Write tex file
    with open('main.tex', 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    # Create Zip
    files_to_zip = ['main.tex']
    
    # Add images
    img_files = [
        'fig1_architecture.png',
        'fig2_simulation_scenario.png',
        'fig3_potential_field.png',
        'fig4_attention_weights.png',
        'fig5_trajectory_comparison.png'
    ]
    
    for img in img_files:
        if os.path.exists(img):
            files_to_zip.append(img)
        else:
            print(f"Warning: Image {img} not found.")

    zip_filename = 'overleaf_submission.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file)
            
    print(f"Successfully created {zip_filename} with {len(files_to_zip)} files.")

if __name__ == "__main__":
    create_zip_package()

