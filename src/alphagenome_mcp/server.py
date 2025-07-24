#!/usr/bin/env python3
"""AlphaGenome MCP Server - Interface for Google DeepMind's AlphaGenome genomics predictions."""

import os
from pathlib import Path
from typing import Any

import matplotlib
import typer
from alphagenome.data import genome
from alphagenome.models import dna_client
from eliot import start_action
from fastmcp import FastMCP
from pydantic import BaseModel, Field

matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3001"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
DEFAULT_OUTPUT_DIR = os.getenv("MCP_OUTPUT_DIR", "gget_output")

class SequencePredictionRequest(BaseModel):
    """Request for sequence prediction."""
    sequence: str = Field(description="DNA sequence to predict (must contain only ACGTN)")
    requested_outputs: list[str] = Field(description="List of output types (e.g., ['RNA_SEQ', 'DNASE'])")
    ontology_terms: list[str] | None = Field(default=None, description="Ontology terms for tissue/cell types (e.g., ['UBERON:0002048'])")
    organism: str = Field(default="HOMO_SAPIENS", description="Organism (HOMO_SAPIENS or MUS_MUSCULUS)")

class IntervalPredictionRequest(BaseModel):
    """Request for interval prediction."""
    chromosome: str = Field(description="Chromosome name (e.g., 'chr1')")
    start: int = Field(description="Start position (0-based)")
    end: int = Field(description="End position (0-based, exclusive)")
    strand: str = Field(default="POSITIVE", description="Strand (POSITIVE or NEGATIVE)")
    requested_outputs: list[str] = Field(description="List of output types")
    ontology_terms: list[str] | None = Field(default=None, description="Ontology terms for tissue/cell types")
    organism: str = Field(default="HOMO_SAPIENS", description="Organism")

class VariantPredictionRequest(BaseModel):
    """Request for variant prediction."""
    chromosome: str = Field(description="Chromosome name for both interval and variant")
    interval_start: int = Field(description="Interval start position (0-based)")
    interval_end: int = Field(description="Interval end position (0-based, exclusive)")
    variant_position: int = Field(description="Variant position (1-based)")
    reference_bases: str = Field(description="Reference bases at variant position")
    alternate_bases: str = Field(description="Alternate bases for the variant")
    requested_outputs: list[str] = Field(description="List of output types")
    ontology_terms: list[str] | None = Field(default=None, description="Ontology terms")
    organism: str = Field(default="HOMO_SAPIENS", description="Organism")

class VariantScoringRequest(BaseModel):
    """Request for variant scoring."""
    chromosome: str = Field(description="Chromosome name")
    interval_start: int = Field(description="Interval start position (0-based)")
    interval_end: int = Field(description="Interval end position (0-based, exclusive)")
    variant_position: int = Field(description="Variant position (1-based)")
    reference_bases: str = Field(description="Reference bases")
    alternate_bases: str = Field(description="Alternate bases")
    variant_scorers: list[str] | None = Field(default=None, description="Variant scorer names (use recommended if None)")
    organism: str = Field(default="HOMO_SAPIENS", description="Organism")

class VisualizationRequest(BaseModel):
    """Request for prediction visualization."""
    plot_type: str = Field(description="Type of plot (tracks, variant_comparison, contact_map)")
    title: str | None = Field(default=None, description="Plot title")
    width: int = Field(default=12, description="Figure width in inches")
    height: int = Field(default=8, description="Figure height in inches")

class PredictionResult(BaseModel):
    """Result from a prediction."""
    outputs: dict[str, Any] = Field(description="Prediction outputs by type")
    metadata: dict[str, Any] = Field(description="Metadata about the prediction")
    interval: dict[str, Any] | None = Field(default=None, description="Genomic interval if applicable")

class VariantResult(BaseModel):
    """Result from variant prediction."""
    reference: dict[str, Any] = Field(description="Reference predictions")
    alternate: dict[str, Any] = Field(description="Alternate predictions")
    variant: dict[str, Any] = Field(description="Variant information")
    interval: dict[str, Any] = Field(description="Genomic interval")

class ScoringResult(BaseModel):
    """Result from variant scoring."""
    scores: list[dict[str, Any]] = Field(description="Scoring results from different scorers")
    variant: dict[str, Any] = Field(description="Variant information")
    interval: dict[str, Any] = Field(description="Genomic interval")

class VisualizationResult(BaseModel):
    """Result from visualization."""
    image_data: str = Field(description="Base64 encoded PNG image")
    image_path: str | None = Field(default=None, description="Path to saved image file")
    plot_info: dict[str, Any] = Field(description="Information about the plot")

class AlphaGenomeMCP(FastMCP):
    """AlphaGenome MCP Server with genomic prediction tools."""

    def __init__(
        self,
        name: str = "AlphaGenome MCP Server",
        prefix: str = "alphagenome_",
        output_dir: str | None = None,
        **kwargs
    ):
        """Initialize the AlphaGenome tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)

        self.prefix = prefix
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # Lazy initialization

        # Register our tools
        self._register_alphagenome_tools()

    def _get_client(self, api_key: str) -> dna_client.DnaClient:
        """Get or create AlphaGenome client."""
        if self._client is None:
            with start_action(action_type="create_alphagenome_client") as action:
                try:
                    self._client = dna_client.create(api_key)
                    action.add_success_fields(client_created=True)
                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Failed to create AlphaGenome client: {e}")
        return self._client

    def _register_alphagenome_tools(self):
        """Register AlphaGenome-specific tools."""

        @self.tool(name=f"{self.prefix}predict_sequence", description="""
        Generate AlphaGenome predictions for a DNA sequence.

        Supports sequences of length 2KB, 16KB, 100KB, 500KB, or 1MB.
        Available output types: RNA_SEQ, CAGE, DNASE, ATAC, CHIP_HISTONE, CHIP_TF,
        SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS, CONTACT_MAPS, PROCAP.

        Use ontology terms like 'UBERON:0002048' (lung) or 'UBERON:0000955' (brain)
        to filter predictions to specific tissues/cell types.
        """)
        def predict_sequence(api_key: str, request: SequencePredictionRequest) -> PredictionResult:
            with start_action(action_type="predict_sequence", sequence_length=len(request.sequence)) as action:
                try:
                    client = self._get_client(api_key)

                    # Convert string output types to enum
                    output_types = [getattr(dna_client.OutputType, ot) for ot in request.requested_outputs]
                    organism = getattr(dna_client.Organism, request.organism)

                    # Make prediction
                    output = client.predict_sequence(
                        sequence=request.sequence,
                        organism=organism,
                        requested_outputs=output_types,
                        ontology_terms=request.ontology_terms
                    )

                    # Convert output to serializable format
                    result_outputs = {}
                    for output_type in request.requested_outputs:
                        attr_name = output_type.lower()
                        if hasattr(output, attr_name):
                            track_data = getattr(output, attr_name)
                            if track_data is not None:
                                result_outputs[output_type] = {
                                    'values_shape': track_data.values.shape,
                                    'metadata_count': len(track_data.metadata),
                                    'interval': {
                                        'chromosome': track_data.interval.chromosome,
                                        'start': track_data.interval.start,
                                        'end': track_data.interval.end,
                                        'strand': track_data.interval.strand.name
                                    } if track_data.interval else None
                                }

                    result = PredictionResult(
                        outputs=result_outputs,
                        metadata={
                            'sequence_length': len(request.sequence),
                            'organism': request.organism,
                            'output_types': request.requested_outputs,
                            'ontology_terms': request.ontology_terms
                        }
                    )

                    action.add_success_fields(outputs_generated=len(result_outputs))
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Sequence prediction failed: {e}")

        @self.tool(name=f"{self.prefix}predict_interval", description="""
        Generate AlphaGenome predictions for a genomic interval.

        Specify chromosome, start, and end positions to predict genomic outputs
        for a specific region. The interval will be automatically resized to
        the nearest supported sequence length if needed.
        """)
        def predict_interval(api_key: str, request: IntervalPredictionRequest) -> PredictionResult:
            with start_action(action_type="predict_interval",
                            chromosome=request.chromosome,
                            start=request.start,
                            end=request.end) as action:
                try:
                    client = self._get_client(api_key)

                    # Create interval
                    strand = getattr(genome.Strand, request.strand)
                    interval = genome.Interval(
                        chromosome=request.chromosome,
                        start=request.start,
                        end=request.end,
                        strand=strand
                    )

                    # Convert parameters
                    output_types = [getattr(dna_client.OutputType, ot) for ot in request.requested_outputs]
                    organism = getattr(dna_client.Organism, request.organism)

                    # Make prediction
                    output = client.predict_interval(
                        interval=interval,
                        organism=organism,
                        requested_outputs=output_types,
                        ontology_terms=request.ontology_terms
                    )

                    # Convert to serializable format
                    result_outputs = {}
                    for output_type in request.requested_outputs:
                        attr_name = output_type.lower()
                        if hasattr(output, attr_name):
                            track_data = getattr(output, attr_name)
                            if track_data is not None:
                                result_outputs[output_type] = {
                                    'values_shape': track_data.values.shape,
                                    'metadata_count': len(track_data.metadata),
                                    'interval': {
                                        'chromosome': track_data.interval.chromosome,
                                        'start': track_data.interval.start,
                                        'end': track_data.interval.end,
                                        'strand': track_data.interval.strand.name
                                    }
                                }

                    result = PredictionResult(
                        outputs=result_outputs,
                        metadata={
                            'organism': request.organism,
                            'output_types': request.requested_outputs,
                            'ontology_terms': request.ontology_terms
                        },
                        interval={
                            'chromosome': interval.chromosome,
                            'start': interval.start,
                            'end': interval.end,
                            'strand': interval.strand.name,
                            'width': interval.width
                        }
                    )

                    action.add_success_fields(outputs_generated=len(result_outputs), interval_width=interval.width)
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Interval prediction failed: {e}")

        @self.tool(name=f"{self.prefix}predict_variant", description="""
        Generate AlphaGenome predictions for a genomic variant.

        Compares predictions between reference and alternate alleles to assess
        the functional impact of a genetic variant. Returns both reference
        and alternate predictions for comparison.
        """)
        def predict_variant(api_key: str, request: VariantPredictionRequest) -> VariantResult:
            with start_action(action_type="predict_variant",
                            chromosome=request.chromosome,
                            variant_position=request.variant_position) as action:
                try:
                    client = self._get_client(api_key)

                    # Create interval and variant
                    interval = genome.Interval(
                        chromosome=request.chromosome,
                        start=request.interval_start,
                        end=request.interval_end
                    )

                    variant = genome.Variant(
                        chromosome=request.chromosome,
                        position=request.variant_position,
                        reference_bases=request.reference_bases,
                        alternate_bases=request.alternate_bases
                    )

                    # Convert parameters
                    output_types = [getattr(dna_client.OutputType, ot) for ot in request.requested_outputs]
                    organism = getattr(dna_client.Organism, request.organism)

                    # Make prediction
                    variant_output = client.predict_variant(
                        interval=interval,
                        variant=variant,
                        organism=organism,
                        requested_outputs=output_types,
                        ontology_terms=request.ontology_terms
                    )

                    # Convert to serializable format
                    def convert_output(output_obj):
                        result = {}
                        for output_type in request.requested_outputs:
                            attr_name = output_type.lower()
                            if hasattr(output_obj, attr_name):
                                track_data = getattr(output_obj, attr_name)
                                if track_data is not None:
                                    result[output_type] = {
                                        'values_shape': track_data.values.shape,
                                        'metadata_count': len(track_data.metadata)
                                    }
                        return result

                    result = VariantResult(
                        reference=convert_output(variant_output.reference),
                        alternate=convert_output(variant_output.alternate),
                        variant={
                            'chromosome': variant.chromosome,
                            'position': variant.position,
                            'reference_bases': variant.reference_bases,
                            'alternate_bases': variant.alternate_bases,
                            'is_snv': variant.is_snv
                        },
                        interval={
                            'chromosome': interval.chromosome,
                            'start': interval.start,
                            'end': interval.end,
                            'width': interval.width
                        }
                    )

                    action.add_success_fields(
                        reference_outputs=len(result.reference),
                        alternate_outputs=len(result.alternate)
                    )
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Variant prediction failed: {e}")

        @self.tool(name=f"{self.prefix}score_variant", description="""
        Score a genomic variant using AlphaGenome variant scorers.

        Applies different scoring methods to quantify the predicted impact
        of a genetic variant. Returns scores from multiple scorers for
        comprehensive variant assessment.
        """)
        def score_variant(api_key: str, request: VariantScoringRequest) -> ScoringResult:
            with start_action(action_type="score_variant",
                            chromosome=request.chromosome,
                            variant_position=request.variant_position) as action:
                try:
                    client = self._get_client(api_key)

                    # Create interval and variant
                    interval = genome.Interval(
                        chromosome=request.chromosome,
                        start=request.interval_start,
                        end=request.interval_end
                    )

                    variant = genome.Variant(
                        chromosome=request.chromosome,
                        position=request.variant_position,
                        reference_bases=request.reference_bases,
                        alternate_bases=request.alternate_bases
                    )

                    # Get organism and scorers
                    organism = getattr(dna_client.Organism, request.organism)

                    if request.variant_scorers:
                        # Convert scorer names to objects
                        scorer_objects = []
                        for _scorer_name in request.variant_scorers:
                            # This is a simplified approach - in practice you'd need
                            # proper mapping from names to scorer objects
                            raise NotImplementedError("Custom scorer selection not yet implemented")
                    else:
                        # Use recommended scorers
                        scorer_objects = None

                    # Score variant
                    scores = client.score_variant(
                        interval=interval,
                        variant=variant,
                        variant_scorers=scorer_objects,
                        organism=organism
                    )

                    # Convert AnnData objects to serializable format
                    score_results = []
                    for i, score_data in enumerate(scores):
                        score_info = {
                            'scorer_index': i,
                            'shape': score_data.shape,
                            'n_obs': score_data.n_obs,
                            'n_vars': score_data.n_vars,
                            'obs_keys': list(score_data.obs.keys()),
                            'var_keys': list(score_data.var.keys()),
                            'uns_keys': list(score_data.uns.keys())
                        }

                        # Add scorer info if available
                        if 'variant_scorer' in score_data.uns:
                            scorer = score_data.uns['variant_scorer']
                            score_info['scorer_name'] = str(scorer)

                        score_results.append(score_info)

                    result = ScoringResult(
                        scores=score_results,
                        variant={
                            'chromosome': variant.chromosome,
                            'position': variant.position,
                            'reference_bases': variant.reference_bases,
                            'alternate_bases': variant.alternate_bases,
                            'is_snv': variant.is_snv
                        },
                        interval={
                            'chromosome': interval.chromosome,
                            'start': interval.start,
                            'end': interval.end,
                            'width': interval.width
                        }
                    )

                    action.add_success_fields(num_scorers=len(score_results))
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Variant scoring failed: {e}")

        @self.tool(name=f"{self.prefix}get_metadata", description="""
        Get metadata about available AlphaGenome outputs and ontology terms.

        Returns information about supported output types, available tissues/cell types,
        and other model metadata for the specified organism.
        """)
        def get_metadata(api_key: str, organism: str = "HOMO_SAPIENS") -> dict[str, Any]:
            with start_action(action_type="get_metadata", organism=organism) as action:
                try:
                    client = self._get_client(api_key)
                    organism_enum = getattr(dna_client.Organism, organism)

                    metadata = client.output_metadata(organism=organism_enum)

                    # Convert to serializable format
                    result = {
                        'organism': organism,
                        'output_types': [ot.name for ot in dna_client.OutputType],
                        'supported_sequence_lengths': list(dna_client.SUPPORTED_SEQUENCE_LENGTHS.values()),
                        'metadata_available': True
                    }

                    # Add metadata details if available
                    if hasattr(metadata, 'atac') and metadata.atac is not None:
                        result['atac_tracks'] = len(metadata.atac)
                    if hasattr(metadata, 'rna_seq') and metadata.rna_seq is not None:
                        result['rna_seq_tracks'] = len(metadata.rna_seq)
                    if hasattr(metadata, 'dnase') and metadata.dnase is not None:
                        result['dnase_tracks'] = len(metadata.dnase)

                    action.add_success_fields(metadata_retrieved=True)
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Failed to get metadata: {e}")

        @self.tool(name=f"{self.prefix}validate_sequence", description="""
        Validate a DNA sequence for AlphaGenome prediction.

        Checks if the sequence contains only valid characters (ACGTN) and
        has a supported length for AlphaGenome predictions.
        """)
        def validate_sequence(sequence: str) -> dict[str, Any]:
            with start_action(action_type="validate_sequence", sequence_length=len(sequence)) as action:
                try:
                    # Check valid characters
                    valid_chars = set('ACGTN')
                    sequence_chars = set(sequence.upper())
                    invalid_chars = sequence_chars - valid_chars

                    # Check length
                    length = len(sequence)
                    supported_lengths = list(dna_client.SUPPORTED_SEQUENCE_LENGTHS.values())
                    is_supported_length = length in supported_lengths

                    # Find closest supported length
                    closest_length = min(supported_lengths, key=lambda x: abs(x - length))

                    result = {
                        'sequence_length': length,
                        'valid_characters': len(invalid_chars) == 0,
                        'invalid_characters': list(invalid_chars) if invalid_chars else [],
                        'supported_length': is_supported_length,
                        'supported_lengths': supported_lengths,
                        'closest_supported_length': closest_length,
                        'valid': len(invalid_chars) == 0 and is_supported_length
                    }

                    action.add_success_fields(
                        valid=result['valid'],
                        sequence_length=length,
                        supported_length=is_supported_length
                    )
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Sequence validation failed: {e}")

        @self.tool(name=f"{self.prefix}get_supported_outputs", description="""
        Get list of supported AlphaGenome output types.

        Returns all available output types that can be requested from AlphaGenome,
        along with descriptions of what each output represents.
        """)
        def get_supported_outputs() -> dict[str, Any]:
            with start_action(action_type="get_supported_outputs") as action:
                try:
                    output_info = {}
                    for output_type in dna_client.OutputType:
                        output_info[output_type.name] = {
                            'name': output_type.name,
                            'description': output_type.__doc__ or "Genomic output type"
                        }

                    # Add detailed descriptions
                    descriptions = {
                        'ATAC': 'ATAC-seq tracks capturing chromatin accessibility',
                        'CAGE': 'CAGE tracks capturing gene expression at transcription start sites',
                        'DNASE': 'DNase I hypersensitive site tracks capturing chromatin accessibility',
                        'RNA_SEQ': 'RNA sequencing tracks capturing gene expression',
                        'CHIP_HISTONE': 'ChIP-seq tracks capturing histone modifications',
                        'CHIP_TF': 'ChIP-seq tracks capturing transcription factor binding',
                        'SPLICE_SITES': 'Splice site tracks capturing donor and acceptor splice sites',
                        'SPLICE_SITE_USAGE': 'Splice site usage tracks',
                        'SPLICE_JUNCTIONS': 'Splice junction tracks from RNA-seq',
                        'CONTACT_MAPS': 'Contact map tracks capturing 3D chromatin interactions',
                        'PROCAP': 'Precision Run-On sequencing and capping tracks'
                    }

                    for output_name, desc in descriptions.items():
                        if output_name in output_info:
                            output_info[output_name]['description'] = desc

                    result = {
                        'output_types': output_info,
                        'count': len(output_info)
                    }

                    action.add_success_fields(output_types_count=len(output_info))
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Failed to get supported outputs: {e}")

        @self.tool(name=f"{self.prefix}get_supported_organisms", description="""
        Get list of supported organisms for AlphaGenome predictions.

        Returns all organisms that AlphaGenome can make predictions for.
        """)
        def get_supported_organisms() -> dict[str, Any]:
            with start_action(action_type="get_supported_organisms") as action:
                try:
                    organisms = {}
                    for organism in dna_client.Organism:
                        organisms[organism.name] = {
                            'name': organism.name,
                            'value': organism.value,
                            'description': f"Organism: {organism.name.replace('_', ' ').title()}"
                        }

                    result = {
                        'organisms': organisms,
                        'count': len(organisms)
                    }

                    action.add_success_fields(organisms_count=len(organisms))
                    return result

                except Exception as e:
                    action.add_error_fields(error=str(e))
                    raise ValueError(f"Failed to get supported organisms: {e}")

# Initialize the AlphaGenome MCP server
mcp = None

# Create typer app
app = typer.Typer(help="AlphaGenome MCP Server - Interface for Google DeepMind's AlphaGenome genomics predictions")

@app.command("run")
def cli_app(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
    transport: str = typer.Option("streamable-http", "--transport", help="Transport type"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Output directory for local files")
) -> None:
    """Run the MCP server with specified transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport=transport, host=host, port=port)

@app.command("stdio")
def cli_app_stdio(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Output directory for local files")
) -> None:
    """Run the MCP server with stdio transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport="stdio")

@app.command("sse")
def cli_app_sse(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Output directory for local files")
) -> None:
    """Run the MCP server with SSE transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport="sse", host=host, port=port)

# Standalone CLI functions for direct script access
def cli_app_run() -> None:
    """Standalone function for alphagenome-mcp-run script."""
    mcp = AlphaGenomeMCP(output_dir=DEFAULT_OUTPUT_DIR)
    mcp.run(transport="streamable-http", host=DEFAULT_HOST, port=DEFAULT_PORT)

def cli_app_stdio_standalone() -> None:
    """Standalone function for alphagenome-mcp-stdio script."""
    mcp = AlphaGenomeMCP(output_dir=DEFAULT_OUTPUT_DIR)
    mcp.run(transport="stdio")

def cli_app_sse_standalone() -> None:
    """Standalone function for alphagenome-mcp-sse script."""
    mcp = AlphaGenomeMCP(output_dir=DEFAULT_OUTPUT_DIR)
    mcp.run(transport="sse", host=DEFAULT_HOST, port=DEFAULT_PORT)

if __name__ == "__main__":
    from pycomfort.logging import to_nice_file, to_nice_stdout

    to_nice_stdout()
    # Determine project root and logs directory
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define log file paths
    json_log_path = log_dir / "mcp_server.log.json"
    rendered_log_path = log_dir / "mcp_server.log"

    # Configure file logging
    to_nice_file(output_file=json_log_path, rendered_file=rendered_log_path)
    app()
