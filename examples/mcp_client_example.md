# AlphaGenome MCP Client Usage Examples

This document shows how to use the AlphaGenome MCP server with different MCP clients.

## Setup

1. Install the AlphaGenome MCP server:
```bash
uvx install alphagenome-mcp
```

2. Set up your environment variables:
```bash
export ALPHAGENOME_API_KEY="your-api-key-here"
export MCP_OUTPUT_DIR="./alphagenome_output"
```

## Configuration Files

### STDIO Mode (Direct Connection)

Use `mcp-config-stdio.json` for direct communication:

```json
{
  "mcpServers": {
    "alphagenome-mcp": {
      "command": "uvx",
      "args": ["alphagenome-mcp-stdio"],
      "env": {
        "ALPHAGENOME_API_KEY": "your-api-key-here",
        "MCP_OUTPUT_DIR": "./alphagenome_output"
      }
    }
  }
}
```

### Server Mode (HTTP/SSE)

First start the server:
```bash
uvx alphagenome-mcp-sse --host localhost --port 3000
```

Then use `mcp-config-server.json`:

```json
{
  "mcpServers": {
    "alphagenome-mcp": {
      "url": "http://localhost:3000/sse",
      "type": "sse",
      "env": {
        "ALPHAGENOME_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Available MCP Tools

### 1. predict_sequence

Predict genomic features for a DNA sequence.

**Parameters:**
- `sequence` (string): DNA sequence (up to 1MB)
- `output_types` (array): Types of predictions to generate
  - `"gene_expression"`
  - `"chromatin_features"`
  - `"splicing_patterns"`
  - `"contact_maps"`

**Example:**
```
Tool: predict_sequence
Parameters:
{
  "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
  "output_types": ["gene_expression", "chromatin_features"]
}
```

### 2. predict_interval

Predict features for a genomic interval.

**Parameters:**
- `chromosome` (string): Chromosome (e.g., "chr1")
- `start` (integer): Start position (0-based)
- `end` (integer): End position (0-based)
- `genome_build` (string): Genome build (default: "hg38")
- `output_types` (array): Same as predict_sequence

**Example:**
```
Tool: predict_interval
Parameters:
{
  "chromosome": "chr1",
  "start": 1000000,
  "end": 1001000,
  "genome_build": "hg38",
  "output_types": ["gene_expression"]
}
```

### 3. predict_variant

Predict the effect of a genetic variant.

**Parameters:**
- `chromosome` (string): Chromosome
- `position` (integer): Position (1-based)
- `reference` (string): Reference allele
- `alternate` (string): Alternate allele
- `context_length` (integer): Context length around variant
- `output_types` (array): Types of predictions

**Example:**
```
Tool: predict_variant
Parameters:
{
  "chromosome": "chr7",
  "position": 117199644,
  "reference": "C",
  "alternate": "T",
  "context_length": 1000,
  "output_types": ["gene_expression", "splicing_patterns"]
}
```

### 4. score_variant

Score the pathogenicity or functional impact of a variant.

**Parameters:**
- `chromosome` (string): Chromosome
- `position` (integer): Position (1-based)
- `reference` (string): Reference allele
- `alternate` (string): Alternate allele
- `score_types` (array): Types of scores to compute
  - `"pathogenicity"`
  - `"splicing_impact"`
  - `"expression_change"`

**Example:**
```
Tool: score_variant
Parameters:
{
  "chromosome": "chr7",
  "position": 117199644,
  "reference": "C",
  "alternate": "T",
  "score_types": ["pathogenicity", "expression_change"]
}
```

### 5. visualize_predictions

Create visualizations of prediction results.

**Parameters:**
- `prediction_file` (string): Path to prediction results file
- `plot_types` (array): Types of plots to generate
  - `"heatmap"`
  - `"line_plot"`
  - `"scatter_plot"`
  - `"genomic_tracks"`
- `region` (object, optional): Specific region to visualize

**Example:**
```
Tool: visualize_predictions
Parameters:
{
  "prediction_file": "./alphagenome_output/predictions_123.json",
  "plot_types": ["heatmap", "genomic_tracks"],
  "region": {
    "start": 1000000,
    "end": 1001000
  }
}
```

## Usage with Claude/Other MCP Clients

Once configured, you can ask your MCP client to:

1. **Analyze a DNA sequence:**
   "Can you predict gene expression for this sequence: ATGCGATCGATCGATCG..."

2. **Investigate a genomic region:**
   "What are the chromatin features in chr1:1000000-1001000?"

3. **Assess variant impact:**
   "Score the pathogenicity of the variant chr7:117199644:C>T"

4. **Create visualizations:**
   "Visualize the predictions from the last analysis as a heatmap"

## Output Files

All results are saved to the configured output directory:

- `predictions_*.json` - Raw prediction data
- `scores_*.json` - Variant scoring results  
- `visualizations_*.png` - Generated plots
- `metadata_*.json` - Analysis metadata

## Troubleshooting

- **API Key Issues**: Ensure `ALPHAGENOME_API_KEY` is set correctly
- **Large Sequences**: For sequences >100kb, consider using intervals
- **Server Mode**: Ensure the server is running before connecting
- **Dependencies**: All required packages are installed automatically with uvx 