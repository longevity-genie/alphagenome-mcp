# AlphaGenome Common Functionality

This document explains how to use the shared AlphaGenome functionality that's common between the MCP server and CLI tools.

## Overview

The AlphaGenome codebase has been refactored to separate core functionality into reusable components:

- **`core.py`** - Core AlphaGenome client operations and utilities
- **`vcf_processor.py`** - VCF file processing for batch variant annotation
- **`server.py`** - MCP server implementation (uses core functionality)
- **`cli.py`** - Command-line interface for VCF annotation

## Core Components

### AlphaGenomeConfig

Configuration class for all AlphaGenome operations:

```python
from alphagenome_mcp.core import AlphaGenomeConfig

# From environment variables
config = AlphaGenomeConfig.from_env(output_dir="my_output")

# Manual configuration
config = AlphaGenomeConfig(
    api_key="your_api_key",
    output_dir=Path("output"),
    organism="HOMO_SAPIENS",
    requested_outputs=["RNA_SEQ", "DNASE", "ATAC"],
    interval_size=100000,
)
```

### AlphaGenomeCore

Core functionality for predictions and analysis:

```python
from alphagenome_mcp.core import AlphaGenomeCore, VariantInfo

core = AlphaGenomeCore(config)

# Single variant analysis
variant = VariantInfo(
    chromosome="chr1",
    position=123456,
    reference_bases="A",
    alternate_bases="T"
)

# Get predictions
predictions = core.predict_variant(variant)

# Get pathogenicity scores
scores = core.score_variant(variant)

# Sequence analysis
results = core.predict_sequence(
    sequence="ATCGATCG" * 256,  # 2KB sequence
    requested_outputs=["RNA_SEQ", "CAGE"]
)
```

### VCF Processing

For batch processing of VCF files:

```python
from alphagenome_mcp.vcf_processor import VCFProcessor, VCFProcessingConfig

# Create configuration
config = VCFProcessingConfig.from_args(
    vcf_file=Path("variants.vcf"),
    output_dir=Path("output"),
    organism="HOMO_SAPIENS",
    outputs=["RNA_SEQ", "DNASE"],
    batch_size=50,
    max_variants=1000,
)

# Process VCF file
processor = VCFProcessor(config)
summary = processor.process_vcf_file(Path("variants.vcf"))
```

## Available Output Types

- **RNA_SEQ** - RNA sequencing gene expression tracks
- **CAGE** - Cap Analysis Gene Expression (transcription start sites)
- **DNASE** - DNase I hypersensitive sites (chromatin accessibility)
- **ATAC** - ATAC-seq chromatin accessibility
- **CHIP_HISTONE** - ChIP-seq histone modification tracks
- **CHIP_TF** - ChIP-seq transcription factor binding sites
- **SPLICE_SITES** - Splice donor and acceptor site predictions
- **SPLICE_SITE_USAGE** - Quantitative splice site usage
- **SPLICE_JUNCTIONS** - RNA-seq splice junction tracks
- **CONTACT_MAPS** - 3D chromatin interaction contact maps
- **PROCAP** - Precision Run-On sequencing and capping

## Supported Organisms

- **HOMO_SAPIENS** - Human (GRCh38/hg38)
- **MUS_MUSCULUS** - Mouse (GRCm39/mm39)

## CLI Usage

### VCF Annotation

```bash
# Basic annotation
python -m alphagenome_mcp.cli annotate-vcf variants.vcf

# Custom output types
python -m alphagenome_mcp.cli annotate-vcf variants.vcf -t RNA_SEQ -t DNASE -t ATAC

# Process subset for testing
python -m alphagenome_mcp.cli annotate-vcf variants.vcf --max-variants 10

# Scoring only (faster)
python -m alphagenome_mcp.cli annotate-vcf variants.vcf --no-predictions --scoring

# Mouse variants
python -m alphagenome_mcp.cli annotate-vcf mouse_variants.vcf --organism MUS_MUSCULUS
```

### Single Variant Testing

```bash
# Test a SNV
python -m alphagenome_mcp.cli test-variant chr1 123456 A T

# With specific outputs
python -m alphagenome_mcp.cli test-variant chr1 123456 A T -t RNA_SEQ -t DNASE

# Test insertion
python -m alphagenome_mcp.cli test-variant chr2 234567 - ATCG

# Test deletion
python -m alphagenome_mcp.cli test-variant chr3 345678 ATG -
```

### Utility Commands

```bash
# List available output types
python -m alphagenome_mcp.cli list-outputs

# Check configuration and API connectivity
python -m alphagenome_mcp.cli check-config
```

## MCP Server Usage

The MCP server now uses the same core functionality:

```python
from alphagenome_mcp import AlphaGenomeMCP, AlphaGenomeConfig

# Custom configuration
config = AlphaGenomeConfig.from_env(output_dir="mcp_output")
mcp = AlphaGenomeMCP(config=config)

# Run server
mcp.run(transport="stdio")
```

## Output Files

All operations generate multiple output formats:

- **NPZ files** - Compressed NumPy arrays for Python analysis
- **Parquet files** - Apache Arrow format for modern data workflows  
- **PNG files** - Auto-generated visualization plots
- **JSON files** - Comprehensive metadata for reproducibility

## Examples

See the `examples/` directory for complete usage examples:

- `examples/cli_example.py` - Core functionality demonstration
- `examples/vcf_example.py` - VCF processing example

## Configuration

### Environment Variables

```bash
export ALPHA_GENOME_API_KEY="your_api_key_here"
```

### API Key Sources

1. `--api-key` command line flag
2. `ALPHA_GENOME_API_KEY` environment variable
3. Configuration file (if implemented)

## Integration Patterns

### Using Core in Your Own Scripts

```python
from alphagenome_mcp.core import AlphaGenomeCore, AlphaGenomeConfig, VariantInfo

# Setup
config = AlphaGenomeConfig.from_env()
core = AlphaGenomeCore(config)

# Process your variants
for variant_data in your_variant_source:
    variant = VariantInfo(
        chromosome=variant_data["chr"],
        position=variant_data["pos"],
        reference_bases=variant_data["ref"],
        alternate_bases=variant_data["alt"],
    )
    
    # Get predictions
    results = core.predict_variant(variant)
    
    # Process results...
```

### Custom VCF Processing

```python
from alphagenome_mcp.vcf_processor import VCFProcessor, VCFProcessingConfig

# Custom processing workflow
config = VCFProcessingConfig(
    alphagenome_config=your_ag_config,
    batch_size=100,
    include_predictions=True,
    include_scoring=False,
)

processor = VCFProcessor(config)

# Process in custom batches
for variant_batch in your_custom_batching:
    results = processor.process_variants_batch(variant_batch)
    # Handle results...
```

## Error Handling

The system includes comprehensive error handling:

- Sequence validation with detailed error messages
- Graceful handling of API failures
- Partial results preservation for batch processing
- Detailed logging via Eliot structured logging

## Performance Considerations

- **Batch Processing** - Process variants in configurable batch sizes
- **Parallel Processing** - Multiple variants processed concurrently
- **Caching** - Client connections are cached and reused
- **Output Formats** - Multiple formats generated simultaneously
- **Memory Management** - Large results written to files rather than kept in memory