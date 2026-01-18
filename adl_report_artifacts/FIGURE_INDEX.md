### ADL report artifacts

Figures:
- `figures/logit_lens_token_frequency_pos1.png` (frequency curves with similarity metrics)
- `figures/logit_lens_side_by_side_pos1.png` (side-by-side comparison with annotations)
- `figures/logit_lens_grouped_bars_pos1.png` (grouped bar comparison highlighting similarities)
- `figures/logit_lens_similarity_heatmap_pos1.png` (heatmap showing token overlap)
- `figures/logit_lens_token_frequency_pos2.png` (frequency curves with similarity metrics)
- `figures/logit_lens_side_by_side_pos2.png` (side-by-side comparison with annotations)
- `figures/logit_lens_grouped_bars_pos2.png` (grouped bar comparison highlighting similarities)
- `figures/logit_lens_similarity_heatmap_pos2.png` (heatmap showing token overlap)

Tables:
- `tables/logit_lens_top_tokens_table.tex`
- `tables/logit_lens_top_tokens_table.csv`

Notes:
- **Frequency curves**: Show token rank vs count with Jaccard similarity and overlap metrics.
- **Side-by-side plots**: Horizontal bar charts with similarity annotations (pedagogically useful).
- **Grouped bar charts**: Direct comparison highlighting when models are similar (green connecting lines).
- **Similarity heatmap**: Visual representation of token overlap between variants.
- **LaTeX table**: Lists the most frequent top-1 tokens per variant/position.

Key insight: Plots are designed to highlight when base and fine-tuned models show
similar behavior (lack of significant differences), with statistical metrics displayed.