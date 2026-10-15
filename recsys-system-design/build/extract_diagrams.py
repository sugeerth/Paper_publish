"""Extract ASCII architecture diagrams (fenced code blocks) from atlas chapters into one Markdown book."""
import re, sys, pathlib
books = [("Recommender Systems", "recsys-system-design"), ("RAG", "rag-architectures"),
         ("LLM Evaluation", "eval-architectures"), ("Closed-Loop AI Systems", "closed-loop-architectures"),
         ("AI Agents", "agent-architectures")]
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
total = 0
for bi, (name, d) in enumerate(books, 1):
    parts = [f"# {bi}. {name} — Architecture Diagrams\n"]
    for f in sorted((root / d / "chapters").glob("*.md"), key=lambda p: p.name):
        text = f.read_text(errors="ignore")
        h1 = re.search(r"^# (.+)$", text, re.M)
        blocks, heading, caption, lines, i = [], "", "", text.split("\n"), 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("## ") or ln.startswith("### "):
                heading, caption = ln.lstrip("# ").strip(), ""
            elif ln.strip().startswith("**") and ln.strip().endswith("**"):
                caption = ln.strip().strip("*")
            elif ln.startswith("```"):
                j = i + 1
                while j < len(lines) and not lines[j].startswith("```"):
                    j += 1
                body = "\n".join(lines[i + 1:j])
                arrows = len(re.findall(r"-->|->|<-|v\b|\^|\|", body))
                boxes = len(re.findall(r"\+-{2,}|\[[^\]]+\]|\(.+?\)", body))
                if body.count("\n") >= 3 and (arrows >= 3 or boxes >= 2) and "def " not in body and "import " not in body:
                    blocks.append((heading, caption, body))
                i = j
            i += 1
        if blocks:
            parts.append(f"## {h1.group(1) if h1 else f.stem}\n")
            for h, c, b in blocks:
                title = c or h
                if title:
                    parts.append(f"**{title}**\n")
                parts.append("```\n" + b + "\n```\n")
            total += len(blocks)
    (out / f"{bi:02d}-{d}.md").write_text("\n".join(parts))
print("diagrams:", total)
