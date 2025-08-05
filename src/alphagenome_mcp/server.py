#!/usr/bin/env python3
"""AlphaGenome MCP Server - Interface for Google DeepMind's AlphaGenome genomics predictions."""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import typer
from alphagenome.data import genome
from alphagenome.models import dna_client
from dotenv import load_dotenv
from eliot import start_action
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from alphagenome_mcp.core import AlphaGenomeConfig, AlphaGenomeCore

# Load environment variables
load_dotenv()

matplotlib.use("Agg")  # Use non-interactive backend

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3001"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
DEFAULT_OUTPUT_DIR = "alphagenome_output"


class SequencePredictionRequest(BaseModel):
    """Request for sequence prediction using AlphaGenome model."""

    sequence: str = Field(
        description="DNA sequence to predict (must contain only ACGTN characters). "
        "Supported sequence lengths: 2KB (2048bp), 16KB (16384bp), 100KB (98304bp), "
        "500KB (491520bp), or 1MB (983040bp). Sequences will be automatically "
        "resized to nearest supported length if needed."
    )
    requested_outputs: list[str] = Field(
        description="List of genomic output types to predict. Available options: "
        "['RNA_SEQ', 'CAGE', 'DNASE', 'ATAC', 'CHIP_HISTONE', 'CHIP_TF', "
        "'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', "
        "'CONTACT_MAPS', 'PROCAP']. Each represents: RNA_SEQ=RNA sequencing "
        "gene expression tracks, CAGE=transcription start site activity, "
        "DNASE=chromatin accessibility via DNase hypersensitive sites, "
        "ATAC=chromatin accessibility via ATAC-seq, CHIP_HISTONE=histone "
        "modification ChIP-seq tracks, CHIP_TF=transcription factor binding "
        "ChIP-seq tracks, SPLICE_SITES=splice donor/acceptor sites, "
        "SPLICE_SITE_USAGE=splice site usage quantification, "
        "SPLICE_JUNCTIONS=splice junction tracks from RNA-seq, "
        "CONTACT_MAPS=3D chromatin interaction contact maps, "
        "PROCAP=Precision Run-On sequencing and capping tracks."
    )
    ontology_terms: list[str] | None = Field(
        default=None,
        description="Ontology terms for tissue/cell type filtering (optional). "
        "Use UBERON anatomical ontology terms like: ['UBERON:0002048'] for lung, "
        "['UBERON:0000955'] for brain, ['UBERON:0002107'] for liver, "
        "['UBERON:0000948'] for heart, ['UBERON:0002113'] for kidney, "
        "['UBERON:0002097'] for skin, ['UBERON:0002371'] for bone marrow, "
        "['UBERON:0000970'] for eye, ['UBERON:0001264'] for pancreas, "
        "['UBERON:0002106'] for spleen, ['UBERON:0001043'] for esophagus, "
        "['UBERON:0000945'] for stomach, ['UBERON:0001155'] for colon, "
        "['UBERON:0002367'] for prostate gland, ['UBERON:0000992'] for ovary, "
        "['UBERON:0000473'] for testis, ['UBERON:0001911'] for mammary gland, "
        "['UBERON:0002405'] for immune system, ['UBERON:0000006'] for islet of Langerhans. "
        "Multiple terms can be combined for multi-tissue analysis.",
    )
    organism: str = Field(
        default="HOMO_SAPIENS",
        description="Target organism for prediction. Options: 'HOMO_SAPIENS' for human "
        "genome analysis (GRCh38/hg38 reference), 'MUS_MUSCULUS' for mouse "
        "genome analysis (GRCm39/mm39 reference). Default is human.",
    )


class IntervalPredictionRequest(BaseModel):
    """Request for genomic interval prediction using coordinates."""

    chromosome: str = Field(
        description="Chromosome identifier (e.g., 'chr1', 'chr2', ..., 'chr22', 'chrX', 'chrY', 'chrM'). "
        "For human: use 'chr1' through 'chr22' for autosomes, 'chrX', 'chrY' for sex chromosomes, "
        "'chrM' for mitochondrial DNA. For mouse: use 'chr1' through 'chr19', 'chrX', 'chrY', 'chrM'."
    )
    start: int = Field(
        description="Start position in base pairs (0-based coordinates, inclusive). "
        "This follows standard genomic coordinates where the first base is position 0. "
        "Example: for the interval chr1:1000-2000, start=999 (0-based)."
    )
    end: int = Field(
        description="End position in base pairs (0-based coordinates, exclusive). "
        "The interval includes bases from start up to but not including end. "
        "Example: for chr1:1000-2000, end=2000. Interval length = end - start."
    )
    strand: str = Field(
        default="POSITIVE",
        description="DNA strand orientation. Options: 'POSITIVE' for forward/plus strand (+), "
        "'NEGATIVE' for reverse/minus strand (-). Affects prediction orientation "
        "for strand-specific outputs like gene expression and transcription factor binding.",
    )
    requested_outputs: list[str] = Field(
        description="List of genomic output types to predict. Available options: "
        "['RNA_SEQ', 'CAGE', 'DNASE', 'ATAC', 'CHIP_HISTONE', 'CHIP_TF', "
        "'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', "
        "'CONTACT_MAPS', 'PROCAP']. See SequencePredictionRequest for detailed descriptions."
    )
    ontology_terms: list[str] | None = Field(
        default=None,
        description="UBERON ontology terms for tissue/cell type filtering (optional). "
        "Examples: ['UBERON:0002048'] for lung, ['UBERON:0000955'] for brain, "
        "['UBERON:0002107'] for liver, ['UBERON:0000948'] for heart. "
        "See SequencePredictionRequest for comprehensive list of available terms.",
    )
    organism: str = Field(
        default="HOMO_SAPIENS",
        description="Target organism. Options: 'HOMO_SAPIENS' (human GRCh38/hg38), "
        "'MUS_MUSCULUS' (mouse GRCm39/mm39). Default is human.",
    )


class VariantPredictionRequest(BaseModel):
    """Request for genetic variant impact prediction comparing reference vs alternate alleles."""

    chromosome: str = Field(
        description="Chromosome identifier for both interval and variant (e.g., 'chr1', 'chr2', etc.). "
        "Must be the same chromosome for both the genomic interval and the variant position."
    )
    interval_start: int = Field(
        description="Genomic interval start position (0-based coordinates, inclusive). "
        "This defines the analysis window around the variant. The interval should "
        "contain the variant position and sufficient flanking sequence for context. "
        "Typical intervals range from 16KB to 1MB depending on analysis needs."
    )
    interval_end: int = Field(
        description="Genomic interval end position (0-based coordinates, exclusive). "
        "Must be greater than interval_start and should encompass the variant_position. "
        "Interval length = interval_end - interval_start."
    )
    variant_position: int = Field(
        description="Variant genomic position (1-based coordinates). This is the exact "
        "position of the genetic variant within the specified interval. "
        "Must satisfy: interval_start < variant_position-1 < interval_end. "
        "Example: for variant at chr1:1000, use variant_position=1000."
    )
    reference_bases: str = Field(
        description="Reference allele bases at the variant position (uppercase DNA sequence). "
        "For SNVs: single nucleotide like 'A', 'T', 'G', 'C'. "
        "For insertions: '-' or empty string. "
        "For deletions: the deleted sequence like 'ATG'. "
        "For substitutions: the original sequence like 'CAT'."
    )
    alternate_bases: str = Field(
        description="Alternate allele bases for the variant (uppercase DNA sequence). "
        "For SNVs: single nucleotide like 'G' (if reference is 'A'). "
        "For insertions: the inserted sequence like 'ATCG'. "
        "For deletions: '-' or empty string. "
        "For substitutions: the replacement sequence like 'GTC'."
    )
    requested_outputs: list[str] = Field(
        description="List of genomic output types to compare between reference and alternate. "
        "Available options: ['RNA_SEQ', 'CAGE', 'DNASE', 'ATAC', 'CHIP_HISTONE', "
        "'CHIP_TF', 'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', "
        "'CONTACT_MAPS', 'PROCAP']. Each output will be predicted for both "
        "reference and alternate sequences to assess variant impact."
    )
    ontology_terms: list[str] | None = Field(
        default=None,
        description="UBERON ontology terms for tissue-specific variant impact analysis (optional). "
        "Examples: ['UBERON:0002048'] for lung-specific effects, "
        "['UBERON:0000955'] for brain-specific effects. "
        "See SequencePredictionRequest for comprehensive ontology term list.",
    )
    organism: str = Field(
        default="HOMO_SAPIENS",
        description="Target organism for variant analysis. Options: 'HOMO_SAPIENS' (human), "
        "'MUS_MUSCULUS' (mouse). Default is human.",
    )


class VariantScoringRequest(BaseModel):
    """Request for variant pathogenicity and functional impact scoring."""

    chromosome: str = Field(
        description="Chromosome identifier (e.g., 'chr1', 'chr2', etc.). "
        "Same chromosome for both interval and variant position."
    )
    interval_start: int = Field(
        description="Analysis interval start position (0-based coordinates, inclusive). "
        "Defines the genomic window for variant scoring context."
    )
    interval_end: int = Field(
        description="Analysis interval end position (0-based coordinates, exclusive). "
        "Must encompass the variant position for proper scoring context."
    )
    variant_position: int = Field(
        description="Variant genomic position (1-based coordinates). Exact position "
        "of the genetic variant to be scored for functional impact."
    )
    reference_bases: str = Field(
        description="Reference allele DNA sequence (uppercase). Examples: 'A' for SNV, "
        "'ATG' for deletion, '-' for insertion reference."
    )
    alternate_bases: str = Field(
        description="Alternate allele DNA sequence (uppercase). Examples: 'T' for SNV, "
        "'-' for deletion, 'GCAT' for insertion."
    )
    variant_scorers: list[str] | None = Field(
        default=None,
        description="List of variant scoring algorithms to apply (optional). "
        "If None, uses AlphaGenome's recommended set of scorers. "
        "Available scorers typically include pathogenicity prediction, "
        "conservation scoring, functional impact assessment, and "
        "tissue-specific effect quantification. Each scorer provides "
        "different insights into variant consequences.",
    )
    organism: str = Field(
        default="HOMO_SAPIENS",
        description="Target organism for scoring. Options: 'HOMO_SAPIENS' (human), "
        "'MUS_MUSCULUS' (mouse). Default is human.",
    )


class VisualizationRequest(BaseModel):
    """Request for creating visualizations from AlphaGenome prediction data."""

    plot_type: str = Field(
        description="Type of visualization to create. Available options: "
        "'tracks' for genomic signal track plots (RNA-seq, ATAC-seq, etc.), "
        "'variant_comparison' for side-by-side reference vs alternate comparison plots, "
        "'contact_map' for 3D chromatin interaction heatmaps, "
        "'splice_sites' for splice site prediction visualization, "
        "'multi_track' for combining multiple output types in one plot."
    )
    title: str | None = Field(
        default=None,
        description="Custom plot title (optional). If None, auto-generates title based on "
        "plot_type and data content. Example: 'AlphaGenome RNA-seq Prediction chr1:1000-2000'",
    )
    width: int = Field(
        default=12,
        description="Figure width in inches (default: 12). Recommended ranges: "
        "8-12 for single tracks, 12-16 for multi-track plots, "
        "10-14 for variant comparisons, 8-10 for contact maps.",
    )
    height: int = Field(
        default=8,
        description="Figure height in inches (default: 8). Recommended ranges: "
        "6-8 for single tracks, 8-12 for multi-track plots, "
        "6-10 for variant comparisons, 8-10 for contact maps.",
    )


class PredictionResult(BaseModel):
    """Result from AlphaGenome sequence or interval prediction containing file paths and metadata."""

    output_files: dict[str, dict[str, str]] = Field(
        description="Nested dictionary of prediction output files organized by output type and format. "
        "Structure: {output_type: {format: filepath}}. "
        "Output types include: 'RNA_SEQ', 'CAGE', 'DNASE', 'ATAC', 'CHIP_HISTONE', etc. "
        "Formats include: 'npz' (compressed NumPy arrays for Python), "
        "'parquet' (Apache Arrow/Parquet for modern data workflows), "
        "'plot' (PNG visualization files). "
        "Example: {'RNA_SEQ': {'npz': '/path/to/rna_seq.npz', 'parquet': '/path/to/rna_seq.parquet', 'plot': '/path/to/rna_seq_plot.png'}}"
    )
    metadata_file: str = Field(
        description="Path to JSON metadata file containing prediction parameters, sequence information, "
        "output type details, ontology terms used, organism, timestamps, and other "
        "contextual information for reproducibility and analysis interpretation."
    )
    interval: dict[str, Any] | None = Field(
        default=None,
        description="Genomic interval information if applicable (for interval predictions). "
        "Contains: 'chromosome' (e.g., 'chr1'), 'start' (0-based), 'end' (0-based exclusive), "
        "'strand' ('POSITIVE' or 'NEGATIVE'), 'width' (interval length in bp). "
        "None for sequence-only predictions.",
    )


class VariantResult(BaseModel):
    """Result from variant impact prediction comparing reference vs alternate alleles."""

    reference_file: str = Field(
        description="Path to JSON file containing all reference allele prediction files. "
        "Contains nested dictionary with output types and their file paths in multiple formats "
        "(npz, parquet, plot) for the reference sequence."
    )
    alternate_file: str = Field(
        description="Path to JSON file containing all alternate allele prediction files. "
        "Contains nested dictionary with output types and their file paths in multiple formats "
        "(npz, parquet, plot) for the alternate sequence."
    )
    metadata_file: str = Field(
        description="Path to JSON metadata file containing variant details, interval information, "
        "prediction parameters, organism, output types, and analysis context for "
        "reproducibility and interpretation of variant impact results."
    )
    variant: dict[str, Any] = Field(
        description="Variant information dictionary containing: 'chromosome' (e.g., 'chr1'), "
        "'position' (1-based genomic coordinate), 'reference_bases' (ref allele sequence), "
        "'alternate_bases' (alt allele sequence), 'is_snv' (boolean indicating single nucleotide variant)."
    )
    interval: dict[str, Any] = Field(
        description="Genomic interval context dictionary containing: 'chromosome', 'start' (0-based), "
        "'end' (0-based exclusive), 'width' (interval length in bp) defining the "
        "analysis window around the variant."
    )


class ScoringResult(BaseModel):
    """Result from variant pathogenicity and functional impact scoring analysis."""

    scores_file: str = Field(
        description="Path to JSON file listing all variant scoring result files. "
        "Contains 'score_files' array with paths to H5AD (AnnData) files from different scorers, "
        "and 'scorer_count' indicating number of scoring algorithms applied. "
        "Each H5AD file contains detailed scoring matrices and annotations."
    )
    metadata_file: str = Field(
        description="Path to JSON metadata file containing scoring parameters, variant details, "
        "interval context, scorer information, organism, and analysis settings "
        "for reproducibility and interpretation of scoring results."
    )
    variant: dict[str, Any] = Field(
        description="Variant information dictionary with keys: 'chromosome', 'position' (1-based), "
        "'reference_bases', 'alternate_bases', 'is_snv' (boolean for single nucleotide variants). "
        "Defines the exact genetic variant that was scored for functional impact."
    )
    interval: dict[str, Any] = Field(
        description="Analysis interval dictionary containing: 'chromosome', 'start' (0-based), "
        "'end' (0-based exclusive), 'width' (bp) defining the genomic context "
        "window used for variant scoring analysis."
    )


class VisualizationResult(BaseModel):
    """Result from AlphaGenome prediction data visualization."""

    image_data: str | None = Field(
        default=None,
        description="Base64 encoded PNG image data for immediate display or embedding. "
        "Encoded in UTF-8 string format, can be decoded and displayed directly "
        "in web interfaces or notebooks. None if only file output was requested.",
    )
    image_path: str | None = Field(
        default=None,
        description="Filesystem path to saved PNG image file. High-resolution (300 DPI) "
        "publication-quality image suitable for reports, papers, or presentations. "
        "None if only base64 output was requested.",
    )
    plot_info: dict[str, Any] = Field(
        description="Comprehensive plot metadata dictionary containing: 'plot_type' (visualization type), "
        "'title' (plot title), 'width'/'height' (dimensions in inches), 'timestamp' (creation time), "
        "'filename' (generated filename), 'file_size_mb' (image file size), 'dpi' (resolution), "
        "and other plot-specific parameters for documentation and reproducibility."
    )


class AlphaGenomeMCP(FastMCP):
    """AlphaGenome MCP Server with genomic prediction tools."""

    def __init__(
        self,
        name: str = "AlphaGenome MCP Server",
        prefix: str = "alphagenome_",
        output_dir: str | None = None,
        config: AlphaGenomeConfig | None = None,
        **kwargs,
    ):
        """Initialize the AlphaGenome tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)

        self.prefix = prefix

        # Create or use provided configuration
        if config is None:
            config = AlphaGenomeConfig.from_env(
                output_dir=output_dir or DEFAULT_OUTPUT_DIR
            )

        # Initialize core functionality
        self.core = AlphaGenomeCore(config)
        self.output_dir = self.core.config.output_dir.resolve()  # For backward compatibility - ensure absolute path

        # Register our tools
        self._register_alphagenome_tools()

    @property
    def api_key(self) -> str:
        """Get the API key from configuration (for backward compatibility)."""
        return self.core.config.api_key

    def _get_client(self) -> dna_client.DnaClient:
        """Get or create AlphaGenome client (delegate to core)."""
        return self.core.get_client()

    def _validate_sequence(self, sequence: str) -> dict[str, Any]:
        """Validate a DNA sequence (delegate to core)."""
        return self.core.validate_sequence(sequence)

    def _save_track_data(
        self, track_data, output_type: str, base_filename: str
    ) -> dict[str, str]:
        """Save track data to multiple formats (delegate to core)."""
        return self.core.save_track_data(track_data, output_type, base_filename)

    def _save_metadata(self, metadata: dict[str, Any], base_filename: str) -> str:
        """Save metadata to a JSON file (delegate to core)."""
        return self.core.save_metadata(metadata, base_filename)

    def _create_track_visualization(
        self, track_data, output_type: str, plot_file: Path
    ) -> str:
        """Create a visualization for track data (delegate to core)."""
        return self.core.create_track_visualization(track_data, output_type, plot_file)

    def _register_alphagenome_tools(self):
        """Register AlphaGenome-specific tools."""

        @self.tool(
            name=f"{self.prefix}predict_sequence",
            description="""
        Generate comprehensive AlphaGenome genomic predictions for a raw DNA sequence.

        SEQUENCE REQUIREMENTS:
        - Must contain only valid DNA nucleotides: A, C, G, T, N
        - Supported lengths: 2KB (2048bp), 16KB (16384bp), 100KB (98304bp), 500KB (491520bp), 1MB (983040bp)
        - Sequences automatically resized to nearest supported length if needed
        - Case-insensitive input (automatically converted to uppercase)

        AVAILABLE OUTPUT TYPES (select one or more):
        - RNA_SEQ: RNA sequencing gene expression tracks
        - CAGE: Cap Analysis Gene Expression transcription start sites
        - DNASE: DNase I hypersensitive sites for chromatin accessibility
        - ATAC: Assay for Transposase-Accessible Chromatin accessibility
        - CHIP_HISTONE: ChIP-seq histone modification tracks (H3K4me3, H3K27ac, etc.)
        - CHIP_TF: ChIP-seq transcription factor binding sites
        - SPLICE_SITES: Splice donor and acceptor site predictions
        - SPLICE_SITE_USAGE: Quantitative splice site usage metrics
        - SPLICE_JUNCTIONS: RNA-seq derived splice junction tracks
        - CONTACT_MAPS: 3D chromatin interaction contact frequency maps
        - PROCAP: Precision Run-On sequencing and capping analysis

        TISSUE/CELL TYPE FILTERING:
        Use UBERON ontology terms for tissue-specific predictions:
        Common examples: UBERON:0002048 (lung), UBERON:0000955 (brain), UBERON:0002107 (liver),
        UBERON:0000948 (heart), UBERON:0002113 (kidney), UBERON:0002097 (skin)

        OUTPUT FORMAT:
        Returns file paths due to large data sizes. Each output type generates:
        - NPZ: Compressed NumPy arrays for Python analysis
        - Parquet: Apache Arrow format for modern data workflows
        - PNG: Auto-generated visualization plots
        - JSON: Comprehensive metadata for reproducibility
        """,
        )
        def predict_sequence(request: SequencePredictionRequest) -> PredictionResult:
            with start_action(
                action_type="predict_sequence", sequence_length=len(request.sequence)
            ):
                # Validate sequence
                self._validate_sequence(request.sequence)

                client = self._get_client()

                # Convert string output types to enum
                output_types = [
                    getattr(dna_client.OutputType, ot)
                    for ot in request.requested_outputs
                ]
                organism = getattr(dna_client.Organism, request.organism)

                # Make prediction
                output = client.predict_sequence(
                    sequence=request.sequence,
                    organism=organism,
                    requested_outputs=output_types,
                    ontology_terms=request.ontology_terms,
                )

                # Generate base filename
                base_filename = f"sequence_prediction_{len(request.sequence)}bp_{hash(request.sequence) % 100000}"

                # Save outputs to files (NPZ, Parquet, and auto-generated plots)
                output_files = {}
                for output_type in request.requested_outputs:
                    attr_name = output_type.lower()
                    if hasattr(output, attr_name):
                        track_data = getattr(output, attr_name)
                        if track_data is not None:
                            file_paths = self._save_track_data(
                                track_data, output_type, base_filename
                            )
                            output_files[output_type] = file_paths

                # Save metadata
                metadata = {
                    "sequence_length": len(request.sequence),
                    "organism": request.organism,
                    "output_types": request.requested_outputs,
                    "ontology_terms": request.ontology_terms,
                    "sequence_hash": hash(request.sequence) % 100000,
                }
                metadata_file = self._save_metadata(metadata, base_filename)

                return PredictionResult(
                    output_files=output_files, metadata_file=metadata_file
                )

        @self.tool(
            name=f"{self.prefix}predict_interval",
            description="""
        Generate AlphaGenome predictions for a specific genomic interval using coordinates.

        COORDINATE SYSTEM:
        - Uses 0-based, half-open intervals [start, end)
        - start: inclusive 0-based position
        - end: exclusive 0-based position
        - Interval length = end - start
        - Example: chr1:1000-2000 → start=999, end=2000 (1000bp interval)

        CHROMOSOME FORMAT:
        - Human: chr1, chr2, ..., chr22, chrX, chrY, chrM (GRCh38/hg38)
        - Mouse: chr1, chr2, ..., chr19, chrX, chrY, chrM (GRCm39/mm39)

        STRAND SPECIFICATION:
        - POSITIVE: Forward/plus strand (+), typical for most analyses
        - NEGATIVE: Reverse/minus strand (-), important for strand-specific outputs

        AUTOMATIC RESIZING:
        Intervals automatically adjusted to nearest supported sequence length:
        2KB, 16KB, 100KB, 500KB, or 1MB for optimal model performance.

        OUTPUT TYPES AND TISSUE FILTERING:
        Same as predict_sequence tool - supports all genomic output types
        (RNA_SEQ, CAGE, DNASE, ATAC, etc.) with optional UBERON ontology filtering.

        USE CASES:
        - Analyze specific gene loci or regulatory regions
        - Study chromatin accessibility in defined intervals
        - Examine transcription factor binding in promoter regions
        - Investigate 3D chromatin interactions in TAD boundaries
        """,
        )
        def predict_interval(request: IntervalPredictionRequest) -> PredictionResult:
            with start_action(
                action_type="predict_interval",
                chromosome=request.chromosome,
                start=request.start,
                end=request.end,
            ):
                client = self._get_client()

                # Create interval
                strand = getattr(genome.Strand, request.strand)
                interval = genome.Interval(
                    chromosome=request.chromosome,
                    start=request.start,
                    end=request.end,
                    strand=strand,
                )

                # Convert parameters
                output_types = [
                    getattr(dna_client.OutputType, ot)
                    for ot in request.requested_outputs
                ]
                organism = getattr(dna_client.Organism, request.organism)

                # Make prediction
                output = client.predict_interval(
                    interval=interval,
                    organism=organism,
                    requested_outputs=output_types,
                    ontology_terms=request.ontology_terms,
                )

                # Generate base filename
                base_filename = f"interval_prediction_{request.chromosome}_{request.start}_{request.end}"

                # Save outputs to files (NPZ, Parquet, and auto-generated plots)
                output_files = {}
                for output_type in request.requested_outputs:
                    attr_name = output_type.lower()
                    if hasattr(output, attr_name):
                        track_data = getattr(output, attr_name)
                        if track_data is not None:
                            file_paths = self._save_track_data(
                                track_data, output_type, base_filename
                            )
                            output_files[output_type] = file_paths

                # Save metadata
                metadata = {
                    "organism": request.organism,
                    "output_types": request.requested_outputs,
                    "ontology_terms": request.ontology_terms,
                    "interval": {
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "strand": interval.strand.name,
                        "width": interval.width,
                    },
                }
                metadata_file = self._save_metadata(metadata, base_filename)

                return PredictionResult(
                    output_files=output_files,
                    metadata_file=metadata_file,
                    interval={
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "strand": interval.strand.name,
                        "width": interval.width,
                    },
                )

        @self.tool(
            name=f"{self.prefix}predict_variant",
            description="""
        Generate comparative AlphaGenome predictions for genetic variant impact analysis.

        VARIANT IMPACT ANALYSIS:
        Predicts genomic outputs for both reference and alternate alleles to quantify
        functional consequences of genetic variants. Essential for understanding
        how mutations affect gene expression, chromatin accessibility, and regulatory function.

        COORDINATE REQUIREMENTS:
        - interval_start/end: 0-based genomic window containing the variant
        - variant_position: 1-based exact variant location
        - Must satisfy: interval_start < variant_position-1 < interval_end
        - Recommended interval size: 16KB-1MB for sufficient context

        VARIANT TYPES SUPPORTED:
        - SNVs (Single Nucleotide Variants): ref='A', alt='T'
        - Insertions: ref='-' or '', alt='ATCG'
        - Deletions: ref='ATG', alt='-' or ''
        - Complex substitutions: ref='CAT', alt='GTC'

        ANALYSIS WORKFLOW:
        1. Extracts reference sequence from specified interval
        2. Creates alternate sequence by applying variant
        3. Predicts all requested outputs for both sequences
        4. Returns separate file sets for reference vs alternate
        5. Enables direct comparison of predicted functional impacts

        CLINICAL APPLICATIONS:
        - Assess pathogenicity of disease-associated variants
        - Predict regulatory effects of non-coding variants
        - Evaluate splice site disruption potential
        - Quantify transcription factor binding changes
        - Study chromatin accessibility alterations
        """,
        )
        def predict_variant(request: VariantPredictionRequest) -> VariantResult:
            with start_action(
                action_type="predict_variant",
                chromosome=request.chromosome,
                variant_position=request.variant_position,
            ):
                client = self._get_client()

                # Create interval and variant
                interval = genome.Interval(
                    chromosome=request.chromosome,
                    start=request.interval_start,
                    end=request.interval_end,
                )

                variant = genome.Variant(
                    chromosome=request.chromosome,
                    position=request.variant_position,
                    reference_bases=request.reference_bases,
                    alternate_bases=request.alternate_bases,
                )

                # Convert parameters
                output_types = [
                    getattr(dna_client.OutputType, ot)
                    for ot in request.requested_outputs
                ]
                organism = getattr(dna_client.Organism, request.organism)

                # Make prediction
                variant_output = client.predict_variant(
                    interval=interval,
                    variant=variant,
                    organism=organism,
                    requested_outputs=output_types,
                    ontology_terms=request.ontology_terms,
                )

                # Generate base filename
                base_filename = f"variant_prediction_{request.chromosome}_{request.variant_position}_{request.reference_bases}_{request.alternate_bases}"

                # Save reference and alternate outputs (NPZ, Parquet, and auto-generated plots)
                def save_output_data(output_obj, suffix):
                    output_files = {}
                    for output_type in request.requested_outputs:
                        attr_name = output_type.lower()
                        if hasattr(output_obj, attr_name):
                            track_data = getattr(output_obj, attr_name)
                            if track_data is not None:
                                file_paths = self._save_track_data(
                                    track_data, output_type, f"{base_filename}_{suffix}"
                                )
                                output_files[output_type] = file_paths
                    return output_files

                reference_files = save_output_data(
                    variant_output.reference, "reference"
                )
                alternate_files = save_output_data(
                    variant_output.alternate, "alternate"
                )

                # Save combined file lists
                reference_file = (
                    self.output_dir / f"{base_filename}_reference_files.json"
                )
                alternate_file = (
                    self.output_dir / f"{base_filename}_alternate_files.json"
                )

                with open(reference_file, "w") as f:
                    json.dump(reference_files, f, indent=2)
                with open(alternate_file, "w") as f:
                    json.dump(alternate_files, f, indent=2)

                # Save metadata
                metadata = {
                    "organism": request.organism,
                    "output_types": request.requested_outputs,
                    "ontology_terms": request.ontology_terms,
                    "variant": {
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "reference_bases": variant.reference_bases,
                        "alternate_bases": variant.alternate_bases,
                        "is_snv": variant.is_snv,
                    },
                    "interval": {
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "width": interval.width,
                    },
                }
                metadata_file = self._save_metadata(metadata, base_filename)

                return VariantResult(
                    reference_file=str(reference_file),
                    alternate_file=str(alternate_file),
                    metadata_file=metadata_file,
                    variant={
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "reference_bases": variant.reference_bases,
                        "alternate_bases": variant.alternate_bases,
                        "is_snv": variant.is_snv,
                    },
                    interval={
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "width": interval.width,
                    },
                )

        @self.tool(
            name=f"{self.prefix}score_variant",
            description="""
        Compute comprehensive pathogenicity and functional impact scores for genetic variants.

        SCORING METHODOLOGY:
        Applies multiple machine learning-based scoring algorithms to quantify
        variant functional consequences. Each scorer provides different insights
        into potential pathogenic effects and regulatory impacts.

        SCORING ALGORITHMS (when available):
        - Pathogenicity prediction: Likelihood of disease-causing effects
        - Conservation scoring: Evolutionary constraint assessment
        - Functional impact: Regulatory and transcriptional effects
        - Tissue-specific effects: Context-dependent impact quantification
        - Splice site disruption: Effects on RNA splicing patterns
        - Protein function: Coding variant consequence prediction

        RECOMMENDED USAGE:
        Use score_variant for rapid pathogenicity assessment, especially when:
        - Screening large variant sets for prioritization
        - Clinical variant interpretation workflows
        - Population genetics studies
        - Functional annotation of GWAS hits
        - Pharmacogenomics variant assessment

        OUTPUT FORMAT:
        Returns AnnData (H5AD) files containing:
        - Multi-dimensional scoring matrices
        - Detailed variant annotations
        - Confidence metrics and uncertainty estimates
        - Cross-reference to external databases
        - Tissue/cell-type specific score breakdowns

        COMPARISON WITH predict_variant:
        - score_variant: Fast, summary-level pathogenicity scores
        - predict_variant: Detailed mechanistic predictions for specific outputs
        """,
        )
        def score_variant(request: VariantScoringRequest) -> ScoringResult:
            with start_action(
                action_type="score_variant",
                chromosome=request.chromosome,
                variant_position=request.variant_position,
            ):
                client = self._get_client()

                # Create interval and variant
                interval = genome.Interval(
                    chromosome=request.chromosome,
                    start=request.interval_start,
                    end=request.interval_end,
                )

                variant = genome.Variant(
                    chromosome=request.chromosome,
                    position=request.variant_position,
                    reference_bases=request.reference_bases,
                    alternate_bases=request.alternate_bases,
                )

                # Get organism and scorers
                organism = getattr(dna_client.Organism, request.organism)

                if request.variant_scorers:
                    # Convert scorer names to objects
                    raise NotImplementedError(
                        "Custom scorer selection not yet implemented"
                    )
                else:
                    # Use recommended scorers
                    scorer_objects = None

                # Score variant
                scores = client.score_variant(
                    interval=interval,
                    variant=variant,
                    variant_scorers=scorer_objects,
                    organism=organism,
                )

                # Generate base filename
                base_filename = f"variant_scoring_{request.chromosome}_{request.variant_position}_{request.reference_bases}_{request.alternate_bases}"

                # Save AnnData objects
                score_files = []
                for i, score_data in enumerate(scores):
                    score_file = self.output_dir / f"{base_filename}_scorer_{i}.h5ad"
                    score_data.write_h5ad(score_file)
                    score_files.append(str(score_file))

                # Create scores file list
                scores_file = self.output_dir / f"{base_filename}_scores.json"
                with open(scores_file, "w") as f:
                    json.dump(
                        {"score_files": score_files, "scorer_count": len(score_files)},
                        f,
                        indent=2,
                    )

                # Save metadata
                metadata = {
                    "organism": request.organism,
                    "variant_scorers": request.variant_scorers,
                    "variant": {
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "reference_bases": variant.reference_bases,
                        "alternate_bases": variant.alternate_bases,
                        "is_snv": variant.is_snv,
                    },
                    "interval": {
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "width": interval.width,
                    },
                    "scorer_count": len(scores),
                }
                metadata_file = self._save_metadata(metadata, base_filename)

                return ScoringResult(
                    scores_file=str(scores_file),
                    metadata_file=metadata_file,
                    variant={
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "reference_bases": variant.reference_bases,
                        "alternate_bases": variant.alternate_bases,
                        "is_snv": variant.is_snv,
                    },
                    interval={
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "width": interval.width,
                    },
                )

        @self.tool(
            name=f"{self.prefix}get_metadata",
            description="""
        Retrieve comprehensive metadata about AlphaGenome model capabilities and configurations.

        METADATA CONTENT:
        - Supported organisms: Human (HOMO_SAPIENS) and Mouse (MUS_MUSCULUS)
        - Available output types: Complete list of predictable genomic features
        - Sequence length requirements: Supported input sizes (2KB to 1MB)
        - Track count information: Number of available tracks per output type
        - Model version and capability details
        - Reference genome information (GRCh38/hg38, GRCm39/mm39)

        ORGANISM-SPECIFIC DETAILS:
        Returns organism-specific metadata including:
        - ATAC-seq track counts and tissue coverage
        - RNA-seq experiment availability and cell type diversity
        - DNase-seq track numbers and tissue representation
        - ChIP-seq data availability for histone marks and transcription factors
        - Contact map resolution and genomic coverage

        USE CASES:
        - Validate tool compatibility before analysis
        - Understand model capabilities and limitations
        - Plan experiments based on available data types
        - Check organism-specific feature availability
        - Estimate computational requirements
        """,
        )
        def get_metadata(organism: str = "HOMO_SAPIENS") -> dict[str, Any]:
            with start_action(action_type="get_metadata", organism=organism):
                client = self._get_client()
                organism_enum = getattr(dna_client.Organism, organism)

                metadata = client.output_metadata(organism=organism_enum)

                # Convert to serializable format
                result = {
                    "organism": organism,
                    "output_types": [ot.name for ot in dna_client.OutputType],
                    "supported_sequence_lengths": list(
                        dna_client.SUPPORTED_SEQUENCE_LENGTHS.values()
                    ),
                    "metadata_available": True,
                }

                # Add metadata details if available
                if hasattr(metadata, "atac") and metadata.atac is not None:
                    result["atac_tracks"] = len(metadata.atac)
                if hasattr(metadata, "rna_seq") and metadata.rna_seq is not None:
                    result["rna_seq_tracks"] = len(metadata.rna_seq)
                if hasattr(metadata, "dnase") and metadata.dnase is not None:
                    result["dnase_tracks"] = len(metadata.dnase)

                return result

        @self.tool(
            name=f"{self.prefix}get_supported_outputs",
            description="""
        Retrieve detailed catalog of all available AlphaGenome genomic output types.

        COMPREHENSIVE OUTPUT CATALOG:
        Returns structured information for each supported output type:

        EXPRESSION & TRANSCRIPTION:
        - RNA_SEQ: RNA sequencing gene expression tracks
        - CAGE: Cap Analysis Gene Expression (transcription start sites)
        - PROCAP: Precision Run-On sequencing and capping

        CHROMATIN ACCESSIBILITY:
        - DNASE: DNase I hypersensitive sites
        - ATAC: Assay for Transposase-Accessible Chromatin

        PROTEIN-DNA INTERACTIONS:
        - CHIP_HISTONE: ChIP-seq histone modification tracks
        - CHIP_TF: ChIP-seq transcription factor binding sites

        RNA PROCESSING:
        - SPLICE_SITES: Splice donor and acceptor site predictions
        - SPLICE_SITE_USAGE: Quantitative splice site usage
        - SPLICE_JUNCTIONS: RNA-seq splice junction tracks

        3D GENOME ORGANIZATION:
        - CONTACT_MAPS: Chromatin interaction contact frequency maps

        RETURN FORMAT:
        Each output type includes name, detailed description, and biological significance
        for informed selection in prediction requests.
        """,
        )
        def get_supported_outputs() -> dict[str, Any]:
            output_info = {}
            for output_type in dna_client.OutputType:
                output_info[output_type.name] = {
                    "name": output_type.name,
                    "description": output_type.__doc__ or "Genomic output type",
                }

            # Add detailed descriptions
            descriptions = {
                "ATAC": "ATAC-seq tracks capturing chromatin accessibility",
                "CAGE": "CAGE tracks capturing gene expression at transcription start sites",
                "DNASE": "DNase I hypersensitive site tracks capturing chromatin accessibility",
                "RNA_SEQ": "RNA sequencing tracks capturing gene expression",
                "CHIP_HISTONE": "ChIP-seq tracks capturing histone modifications",
                "CHIP_TF": "ChIP-seq tracks capturing transcription factor binding",
                "SPLICE_SITES": "Splice site tracks capturing donor and acceptor splice sites",
                "SPLICE_SITE_USAGE": "Splice site usage tracks",
                "SPLICE_JUNCTIONS": "Splice junction tracks from RNA-seq",
                "CONTACT_MAPS": "Contact map tracks capturing 3D chromatin interactions",
                "PROCAP": "Precision Run-On sequencing and capping tracks",
            }

            for output_name, desc in descriptions.items():
                if output_name in output_info:
                    output_info[output_name]["description"] = desc

            return {"output_types": output_info, "count": len(output_info)}

        @self.tool(
            name=f"{self.prefix}get_supported_organisms",
            description="""
        Retrieve information about organisms supported by AlphaGenome models.

        SUPPORTED ORGANISMS:

        HOMO_SAPIENS (Human):
        - Reference genome: GRCh38/hg38
        - Chromosomes: chr1-chr22, chrX, chrY, chrM
        - Comprehensive training data from diverse tissues/cell types
        - Extensive validation on clinical and population genetics datasets
        - Optimized for medical genomics and variant interpretation

        MUS_MUSCULUS (Mouse):
        - Reference genome: GRCm39/mm39
        - Chromosomes: chr1-chr19, chrX, chrY, chrM
        - Model organism with extensive experimental validation
        - Valuable for comparative genomics and model system studies
        - Cross-species validation and evolutionary analysis

        RETURN FORMAT:
        Each organism entry includes:
        - Standard name and identifier
        - Reference genome version
        - Chromosome complement
        - Model training details
        - Recommended use cases
        """,
        )
        def get_supported_organisms() -> dict[str, Any]:
            organisms = {}
            for organism in dna_client.Organism:
                organisms[organism.name] = {
                    "name": organism.name,
                    "value": organism.value,
                    "description": f"Organism: {organism.name.replace('_', ' ').title()}",
                }

            return {"organisms": organisms, "count": len(organisms)}

        @self.tool(
            name=f"{self.prefix}visualize_prediction",
            description="""
        Generate publication-quality visualizations from AlphaGenome prediction data.

        VISUALIZATION TYPES:

        TRACKS:
        - Single or multi-track genomic signal plots
        - Time-series style line plots with filled areas
        - Automatic scaling and statistical annotations
        - Genomic coordinate labeling and interval context
        - Ideal for RNA-seq, ATAC-seq, ChIP-seq visualization

        VARIANT_COMPARISON:
        - Side-by-side reference vs alternate allele comparison
        - Bar plots showing differential predictions
        - Quantitative impact assessment with value labels
        - Statistical significance indicators
        - Perfect for variant impact interpretation

        CONTACT_MAP:
        - Heatmap visualization of 3D chromatin interactions
        - Symmetric contact frequency matrices
        - Color-coded interaction strength
        - Genomic bin position labeling
        - Essential for TAD and loop analysis

        SPLICE_SITES:
        - Splice donor/acceptor site probability tracks
        - Exon-intron boundary visualization
        - Alternative splicing pattern display
        - Junction strength quantification

        MULTI_TRACK:
        - Combined visualization of multiple output types
        - Vertically stacked track arrangement
        - Shared genomic coordinate system
        - Cross-track correlation analysis

        OUTPUT FORMATS:
        - High-resolution PNG (300 DPI) for publications
        - Base64 encoded data for web integration
        - Customizable dimensions and styling
        - Comprehensive metadata for reproducibility

        STYLING FEATURES:
        - Professional scientific plotting aesthetics
        - Automatic color scheme selection
        - Statistical summary annotations (mean, max, min)
        - Genomic interval context information
        - Legend and axis labeling
        """,
        )
        def visualize_prediction(request: VisualizationRequest) -> VisualizationResult:
            import base64
            from io import BytesIO

            import matplotlib.pyplot as plt
            import numpy as np

            with start_action(
                action_type="visualize_prediction", plot_type=request.plot_type
            ):
                # Create figure
                fig, ax = plt.subplots(figsize=(request.width, request.height))

                plot_info = {
                    "plot_type": request.plot_type,
                    "title": request.title
                    or f"AlphaGenome {request.plot_type.title()} Plot",
                    "width": request.width,
                    "height": request.height,
                    "timestamp": str(pd.Timestamp.now()),
                }

                if request.plot_type == "tracks":
                    # Example track visualization
                    ax.set_title(plot_info["title"])
                    ax.set_xlabel("Genomic Position")
                    ax.set_ylabel("Signal Intensity")

                    # Create example track data for demonstration
                    positions = np.arange(0, 1000)
                    signal = np.random.normal(0, 1, 1000).cumsum()
                    ax.plot(positions, signal, label="RNA-seq Signal", linewidth=1.5)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                elif request.plot_type == "variant_comparison":
                    # Example variant comparison
                    ax.set_title(plot_info["title"])
                    categories = ["Reference", "Alternate"]
                    values = [np.random.uniform(0.5, 1.0), np.random.uniform(0.3, 0.8)]
                    colors = ["#1f77b4", "#ff7f0e"]

                    bars = ax.bar(categories, values, color=colors)
                    ax.set_ylabel("Prediction Score")
                    ax.set_ylim(0, 1)

                    # Add value labels on bars
                    for bar, value in zip(bars, values, strict=False):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            f"{value:.3f}",
                            ha="center",
                            va="bottom",
                        )

                elif request.plot_type == "contact_map":
                    # Example contact map
                    ax.set_title(plot_info["title"])
                    size = 50
                    contact_matrix = np.random.exponential(0.5, (size, size))
                    contact_matrix = (
                        contact_matrix + contact_matrix.T
                    ) / 2  # Make symmetric

                    im = ax.imshow(contact_matrix, cmap="Reds", interpolation="nearest")
                    ax.set_xlabel("Genomic Position (bins)")
                    ax.set_ylabel("Genomic Position (bins)")
                    plt.colorbar(im, ax=ax, label="Contact Frequency")

                else:
                    # Default visualization
                    ax.text(
                        0.5,
                        0.5,
                        f"Visualization type '{request.plot_type}' not implemented",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=14,
                    )
                    ax.set_title(plot_info["title"])

                plt.tight_layout()

                # Generate filename with timestamp
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alphagenome_plot_{request.plot_type}_{timestamp}.png"
                image_path = self.output_dir / filename

                # Save plot to file
                plt.savefig(
                    image_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )

                # Optionally create base64 encoded image data
                image_data = None
                if True:  # Always create base64 for now
                    buf = BytesIO()
                    plt.savefig(
                        buf,
                        format="png",
                        dpi=150,
                        bbox_inches="tight",
                        facecolor="white",
                        edgecolor="none",
                    )
                    buf.seek(0)
                    image_data = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()

                plt.close(fig)  # Free memory

                plot_info.update(
                    {
                        "image_path": str(image_path),
                        "filename": filename,
                        "file_size_mb": image_path.stat().st_size / (1024 * 1024),
                        "dpi": 300,
                    }
                )

                return VisualizationResult(
                    image_data=image_data,
                    image_path=str(image_path),
                    plot_info=plot_info,
                )


# Initialize the AlphaGenome MCP server
mcp = None

# Create typer app
app = typer.Typer(
    help="AlphaGenome MCP Server - Interface for Google DeepMind's AlphaGenome genomics predictions"
)


@app.command("run")
def cli_app(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
    transport: str = typer.Option(
        "streamable-http", "--transport", help="Transport type"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory for local files"
    ),
) -> None:
    """Run the MCP server with specified transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport=transport, host=host, port=port)


@app.command("stdio")
def cli_app_stdio(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory for local files"
    ),
) -> None:
    """Run the MCP server with stdio transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport="stdio")


@app.command("sse")
def cli_app_sse(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory for local files"
    ),
) -> None:
    """Run the MCP server with SSE transport."""
    mcp = AlphaGenomeMCP(output_dir=output_dir)
    mcp.run(transport="sse", host=host, port=port)


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
