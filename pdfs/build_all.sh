#!/usr/bin/env bash
# Rebuild every PDF in pdfs/ from its chapter sources.
cd "$(dirname "$0")/.."
B=recsys-system-design/build/make_pdf.sh
$B recsys-system-design "The Recommender Systems Design Atlas" "Production recommender architectures from across the industry" pdfs/01-Recommender-Systems-Design-Atlas.pdf &
$B rag-architectures "The RAG Architectures Atlas" "Retrieval-augmented generation systems in production" pdfs/02-RAG-Architectures-Atlas.pdf &
$B eval-architectures "The LLM Evaluation Architectures Atlas" "Benchmarks, harnesses, judges and production evals" pdfs/03-LLM-Evaluation-Architectures-Atlas.pdf &
$B closed-loop-architectures "The Closed-Loop AI Systems Atlas" "Feedback loops, data flywheels and self-improving systems" pdfs/04-Closed-Loop-AI-Systems-Atlas.pdf &
$B agent-architectures "The AI Agent Architectures Atlas" "Designs of production AI agents" pdfs/05-AI-Agent-Architectures-Atlas.pdf &
wait
for c in amazon-l5-applied-scientist:"Amazon L5 Applied Scientist":06-Interview-Amazon-L5-Applied-Scientist \
         airbnb-senior-mle:"Airbnb Senior ML Engineer":07-Interview-Airbnb-Senior-MLE \
         cresta-senior-mle:"Cresta Senior ML Engineer":08-Interview-Cresta-Senior-MLE \
         uipath-principal-applied-scientist:"UiPath Principal Applied Scientist":10-Interview-UiPath-Principal-Applied-Scientist; do
  IFS=: read -r dir title out <<< "$c"
  ls interview-prep/$dir/chapters/*.md >/dev/null 2>&1 && $B interview-prep/$dir "$title" "Onsite interview question bank" pdfs/$out.pdf &
done
ls ml-system-design-diagrams/chapters/*.md >/dev/null 2>&1 && $B ml-system-design-diagrams "The ML System Design Interview Diagram Atlas" "Diagrams compiled from interview portals and engineering blogs" pdfs/11-ML-System-Design-Interview-Diagrams.pdf &
D=$(mktemp -d); mkdir -p $D/chapters; python3 -I recsys-system-design/build/extract_diagrams.py "$PWD" $D/chapters >/dev/null
$B $D "Architecture Diagrams Compendium" "System diagrams extracted from the five architecture atlases" pdfs/09-Architecture-Diagrams-Compendium.pdf &
wait
