# LucusRAG Evaluation Framework

Comprehensive evaluation framework for assessing the retrieval performance of the LucusRAG system.

## Overview

This evaluation suite measures the quality of the RAG system's document retrieval using standard information retrieval metrics across different top-k values.

## Metrics

### 1. **Precision@k**
- **Definition**: Fraction of retrieved documents that are relevant
- **Formula**: `relevant_retrieved / k`
- **Interpretation**: High precision = few irrelevant results

### 2. **Recall@k**
- **Definition**: Fraction of relevant documents that were retrieved
- **Formula**: `relevant_retrieved / total_relevant`
- **Interpretation**: High recall = most relevant docs found

### 3. **F1@k**
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`
- **Interpretation**: Balanced measure of retrieval quality

### 4. **MRR (Mean Reciprocal Rank)**
- **Definition**: Average of reciprocal ranks of first relevant document
- **Formula**: `1 / rank_of_first_relevant`
- **Interpretation**: Measures how quickly relevant results appear

### 5. **NDCG@k (Normalized Discounted Cumulative Gain)**
- **Definition**: Normalized measure that considers ranking quality
- **Interpretation**: Values closer to 1.0 indicate better ranking

## Files

### `test_queries.json`
Test dataset containing:
- Query text
- Category (for stratified analysis)
- Ground truth relevant files
- Ground truth relevant code elements

**Structure:**
```json
{
  "query": "How does X work?",
  "category": "understanding",
  "relevant_files": ["file1.py", "file2.py"],
  "relevant_elements": ["class:X", "method:X.process"]
}
```

### `evaluation_notebook.ipynb`
Interactive Jupyter notebook for:
- Running evaluation experiments
- Computing metrics at different k values
- Generating visualizations
- Analyzing results by category
- Identifying best/worst queries

## Usage

### 1. Setup Environment

```bash
# Ensure you're in the lucusrag conda environment
conda activate lucusrag

# Install additional dependencies for evaluation
pip install matplotlib seaborn jupyter ipykernel
```

### 2. Configure Environment Variables

Make sure your `.env` file contains:
```bash
AST_CACHE_DIR=./ast_cache
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
VOYAGE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### 3. Start Neo4j

```bash
docker-compose up -d
```

### 4. Run Evaluation

```bash
cd evaluation
jupyter notebook evaluation_notebook.ipynb
```

Run all cells in order to:
1. Initialize the RAG system
2. Load test queries
3. Retrieve documents at different k values
4. Calculate metrics
5. Generate visualizations
6. Save results

## Interpreting Results

### Output Files

1. **`evaluation_results.csv`**: Detailed per-query metrics at each k value
2. **`metrics_by_k.csv`**: Aggregated metrics by k value
3. **`evaluation_summary.txt`**: Human-readable summary report
4. **`metrics_vs_k.png`**: Line plot of precision, recall, F1 vs k
5. **`f1_heatmap.png`**: Heatmap of F1 scores by category and k
6. **`precision_distribution.png`**: Box plots of precision distribution

### Key Questions to Answer

1. **What is the optimal k value?**
   - Look for k where F1 score is maximized
   - Consider the precision-recall trade-off

2. **Which query categories perform poorly?**
   - Check the heatmap for low F1 scores
   - Investigate specific queries in those categories

3. **Is the ranking quality good?**
   - High NDCG (>0.7) indicates relevant docs appear early
   - Low NDCG suggests re-ranking may be needed

4. **Are results consistent?**
   - Check box plots for variance
   - High variance suggests unstable retrieval

## Customization

### Adding New Test Queries

Edit `test_queries.json`:
```json
{
  "query": "Your question here",
  "category": "your_category",
  "relevant_files": ["path/to/relevant/file.py"],
  "relevant_elements": ["type:ElementName"]
}
```

### Testing Different K Values

Modify in notebook:
```python
k_values = [1, 3, 5, 10, 15, 20, 30, 50]
```

### Changing Similarity Threshold

The system uses a similarity cutoff of 0.3 by default (defined in `engine.py`). To test different thresholds, you'll need to modify the query engine configuration.

## Performance Benchmarks

Expected performance targets:
- **Precision@5**: >0.6 (at least 60% of top-5 are relevant)
- **Recall@10**: >0.7 (at least 70% of relevant docs in top-10)
- **F1@5**: >0.6 (balanced precision-recall)
- **NDCG@10**: >0.7 (good ranking quality)
- **MRR**: >0.6 (relevant docs appear in top positions)

## Troubleshooting

### Issue: Low recall at all k values
**Cause**: Relevant documents not being retrieved
**Solution**: 
- Check embedding quality
- Verify ground truth is accurate
- Consider query expansion

### Issue: High recall but low precision
**Cause**: Too many irrelevant documents retrieved
**Solution**:
- Increase similarity threshold
- Add re-ranking stage
- Improve embedding model

### Issue: Good metrics but poor user experience
**Cause**: Ranking quality issues (check NDCG)
**Solution**:
- Implement learning-to-rank
- Add metadata filtering
- Use hybrid retrieval (semantic + keyword)

## Future Enhancements

- [ ] A/B testing framework for comparing different configurations
- [ ] Temporal analysis (performance over time)
- [ ] Query difficulty estimation
- [ ] Error analysis dashboard
- [ ] Automatic ground truth generation from code navigation patterns
- [ ] Cross-validation with multiple test sets
- [ ] Integration with CI/CD for continuous evaluation

## Contributing

To add new evaluation metrics:
1. Define the metric function in the notebook
2. Add it to the evaluation loop
3. Include it in visualizations
4. Update this README

## References

- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [NDCG Explained](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [LlamaIndex Evaluation Guide](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)

