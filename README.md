# AlphaGenome MCP Server

[![CI](https://github.com/your-username/alphagenome-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/alphagenome-mcp/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

An MCP (Model Context Protocol) server that provides interface to Google DeepMind's [AlphaGenome](https://github.com/google-deepmind/alphagenome) genomics predictions API.

AlphaGenome is a unifying model for deciphering the regulatory code within DNA sequences, offering multimodal predictions including gene expression, splicing patterns, chromatin features, and contact maps at single base-pair resolution.

## Features

- **Sequence Predictions**: Generate predictions for DNA sequences up to 1MB in length
- **Interval Predictions**: Predict genomic outputs for specific chromosomal regions
- **Variant Analysis**: Assess functional impact of genetic variants by comparing reference vs alternate predictions
- **Variant Scoring**: Quantify variant effects using multiple scoring methods
- **Visualization**: Create publication-quality plots and charts from prediction data
- **Metadata Access**: Retrieve information about available output types and organisms
- **Validation Tools**: Validate DNA sequences and check supported parameters

## Supported Output Types

- **RNA_SEQ**: RNA sequencing tracks capturing gene expression
- **CAGE**: CAGE tracks capturing gene expression at transcription start sites
- **DNASE**: DNase I hypersensitive site tracks capturing chromatin accessibility
- **ATAC**: ATAC-seq tracks capturing chromatin accessibility
- **CHIP_HISTONE**: ChIP-seq tracks capturing histone modifications
- **CHIP_TF**: ChIP-seq tracks capturing transcription factor binding
- **SPLICE_SITES**: Splice site tracks capturing donor and acceptor sites
- **SPLICE_SITE_USAGE**: Splice site usage fraction tracks
- **SPLICE_JUNCTIONS**: Splice junction tracks from RNA-seq
- **CONTACT_MAPS**: Contact map tracks capturing 3D chromatin interactions
- **PROCAP**: Precision Run-On sequencing and capping tracks

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- AlphaGenome API key from [Google DeepMind](https://deepmind.google.com/science/alphagenome)

### Install from source

```bash
git clone https://github.com/your-username/alphagenome-mcp.git
cd alphagenome-mcp
uv sync
```

### Install development dependencies

```bash
uv sync --dev
```

## Usage

### Command Line Interface

The server supports multiple transport protocols:

```bash
# HTTP transport (default)
alphagenome-mcp run --host 0.0.0.0 --port 3001

# stdio transport (for MCP clients)
alphagenome-mcp stdio

# Server-Sent Events transport
alphagenome-mcp sse --host 0.0.0.0 --port 3001
```

### MCP Tools

The server provides the following tools:

#### `alphagenome_predict_sequence`
Generate predictions for a DNA sequence.

```json
{
  "name": "alphagenome_predict_sequence",
  "arguments": {
    "api_key": "your_api_key",
    "request": {
      "sequence": "ATCGATCGATCG...",
      "requested_outputs": ["RNA_SEQ", "DNASE"],
      "ontology_terms": ["UBERON:0002048"],
      "organism": "HOMO_SAPIENS"
    }
  }
}
```

#### `alphagenome_predict_interval`
Generate predictions for a genomic interval.

```json
{
  "name": "alphagenome_predict_interval",
  "arguments": {
    "api_key": "your_api_key",
    "request": {
      "chromosome": "chr1",
      "start": 1000000,
      "end": 1002048,
      "requested_outputs": ["RNA_SEQ"],
      "organism": "HOMO_SAPIENS"
    }
  }
}
```

#### `alphagenome_predict_variant`
Predict effects of genetic variants.

```json
{
  "name": "alphagenome_predict_variant",
  "arguments": {
    "api_key": "your_api_key",
    "request": {
      "chromosome": "chr1",
      "interval_start": 1000000,
      "interval_end": 1002048,
      "variant_position": 1001024,
      "reference_bases": "A",
      "alternate_bases": "T",
      "requested_outputs": ["RNA_SEQ"]
    }
  }
}
```

#### `alphagenome_score_variant`
Score variants using AlphaGenome scorers.

```json
{
  "name": "alphagenome_score_variant",
  "arguments": {
    "api_key": "your_api_key",
    "request": {
      "chromosome": "chr1",
      "interval_start": 1000000,
      "interval_end": 1002048,
      "variant_position": 1001024,
      "reference_bases": "A",
      "alternate_bases": "T"
    }
  }
}
```

#### Utility Tools

- `alphagenome_validate_sequence`: Validate DNA sequence format
- `alphagenome_get_metadata`: Get model metadata for organisms
- `alphagenome_get_supported_outputs`: List available output types
- `alphagenome_get_supported_organisms`: List supported organisms

### Integration with MCP Clients

This server is designed to work with MCP-compatible clients like:

- [Claude Desktop](https://claude.ai/desktop)
- [Cline](https://github.com/clinebot/cline)
- Other MCP-enabled applications

Add the server to your MCP client configuration:

```json
{
  "mcpServers": {
    "alphagenome": {
      "command": "uv",
      "args": ["run", "alphagenome-mcp", "stdio"],
      "cwd": "/path/to/alphagenome-mcp"
    }
  }
}
```

## Development

### Running Tests

```bash
# Run tests that don't require API key (model validation, etc.)
uv run pytest tests/test_server.py::TestPredictionRequestModels tests/test_server.py::TestInternalValidation -v

# Run all tests (requires API key for AlphaGenome API access)
export ALPHA_GENOME_API_KEY="your_api_key"
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=alphagenome_mcp --cov-report=html
```

### GitHub Actions CI

The project includes GitHub Actions CI that:
- Runs basic tests without API key (model validation, file operations)
- Runs integration tests with real AlphaGenome API calls (if API key is available)
- Performs linting, type checking, and builds the package

To enable integration tests in CI, add your AlphaGenome API key as a repository secret:
1. Go to your repository Settings → Secrets and variables → Actions
2. Add a new secret named `ALPHA_GENOME_API_KEY`
3. Set the value to your AlphaGenome API key

The CI will automatically run on pushes and pull requests to `main` and `develop` branches.

### Code Quality

```bash
# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src
```

### Project Structure

```
alphagenome-mcp/
├── src/alphagenome_mcp/
│   ├── __init__.py
│   └── server.py          # Main MCP server implementation
├── tests/
│   ├── __init__.py
│   └── test_server.py     # Test suite
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── pyproject.toml         # Project configuration
├── pytest.ini            # Test configuration
├── ruff.toml             # Linting configuration
└── README.md
```

## API Requirements

To use this server, you need:

1. An AlphaGenome API key from [Google DeepMind](https://deepmind.google.com/science/alphagenome)
2. Accept the AlphaGenome [Terms of Use](https://deepmind.google.com/science/alphagenome/terms)

The API is offered free of charge for non-commercial use, subject to rate limits.

## Supported Sequence Lengths

AlphaGenome supports the following sequence lengths:
- 2KB (2,048 bp)
- 16KB (16,384 bp)  
- 100KB (131,072 bp)
- 500KB (524,288 bp)
- 1MB (1,048,576 bp)

## Supported Organisms

- **HOMO_SAPIENS**: Human (Homo sapiens)
- **MUS_MUSCULUS**: Mouse (Mus musculus)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`uv run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AlphaGenome](https://github.com/google-deepmind/alphagenome) by Google DeepMind
- [Model Context Protocol](https://modelcontextprotocol.io/) specification
- [FastMCP](https://github.com/jlowin/fastmcp) framework

## Citation

If you use AlphaGenome in your research, please cite:

```bibtex
@article{alphagenome,
  title={{AlphaGenome}: advancing regulatory variant effect prediction with a unified {DNA} sequence model},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  year={2025},
  doi={https://doi.org/10.1101/2025.06.25.661532},
  publisher={Cold Spring Harbor Laboratory},
  journal={bioRxiv}
}
```
