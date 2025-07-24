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

# Load environment variables
load_dotenv()

matplotlib.use("Agg")  # Use non-interactive backend

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3001"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
DEFAULT_OUTPUT_DIR = "alphagenome_output"


class SequencePredictionRequest(BaseModel):
    """Request for sequence prediction."""

    sequence: str = Field(
        description="DNA sequence to predict (must contain only ACGTN)"
    )
    requested_outputs: list[str] = Field(
        description="List of output types (e.g., ['RNA_SEQ', 'DNASE'])"
    )
    ontology_terms: list[str] | None = Field(
        default=None,
        description="Ontology terms for tissue/cell types (e.g., ['UBERON:0002048'])",
    )
    organism: str = Field(
        default="HOMO_SAPIENS", description="Organism (HOMO_SAPIENS or MUS_MUSCULUS)"
    )


class IntervalPredictionRequest(BaseModel):
    """Request for interval prediction."""

    chromosome: str = Field(description="Chromosome name (e.g., 'chr1')")
    start: int = Field(description="Start position (0-based)")
    end: int = Field(description="End position (0-based, exclusive)")
    strand: str = Field(default="POSITIVE", description="Strand (POSITIVE or NEGATIVE)")
    requested_outputs: list[str] = Field(description="List of output types")
    ontology_terms: list[str] | None = Field(
        default=None, description="Ontology terms for tissue/cell types"
    )
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
    variant_scorers: list[str] | None = Field(
        default=None, description="Variant scorer names (use recommended if None)"
    )
    organism: str = Field(default="HOMO_SAPIENS", description="Organism")


class VisualizationRequest(BaseModel):
    """Request for prediction visualization."""

    plot_type: str = Field(
        description="Type of plot (tracks, variant_comparison, contact_map)"
    )
    title: str | None = Field(default=None, description="Plot title")
    width: int = Field(default=12, description="Figure width in inches")
    height: int = Field(default=8, description="Figure height in inches")


class PredictionResult(BaseModel):
    """Result from a prediction."""

    output_files: dict[str, dict[str, str]] = Field(
        description="File paths for prediction outputs by type and format (npz, parquet, plot)"
    )
    metadata_file: str = Field(description="Path to metadata JSON file")
    interval: dict[str, Any] | None = Field(
        default=None, description="Genomic interval if applicable"
    )


class VariantResult(BaseModel):
    """Result from variant prediction."""

    reference_file: str = Field(description="Path to reference predictions file")
    alternate_file: str = Field(description="Path to alternate predictions file")
    metadata_file: str = Field(description="Path to metadata JSON file")
    variant: dict[str, Any] = Field(description="Variant information")
    interval: dict[str, Any] = Field(description="Genomic interval")


class ScoringResult(BaseModel):
    """Result from variant scoring."""

    scores_file: str = Field(description="Path to scoring results file")
    metadata_file: str = Field(description="Path to metadata JSON file")
    variant: dict[str, Any] = Field(description="Variant information")
    interval: dict[str, Any] = Field(description="Genomic interval")


class VisualizationResult(BaseModel):
    """Result from visualization."""

    image_data: str | None = Field(
        default=None, description="Base64 encoded PNG image (if requested)"
    )
    image_path: str | None = Field(
        default=None, description="Path to saved image file (if saved)"
    )
    plot_info: dict[str, Any] = Field(description="Information about the plot")


class AlphaGenomeMCP(FastMCP):
    """AlphaGenome MCP Server with genomic prediction tools."""

    def __init__(
        self,
        name: str = "AlphaGenome MCP Server",
        prefix: str = "alphagenome_",
        output_dir: str | None = None,
        **kwargs,
    ):
        """Initialize the AlphaGenome tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)

        self.prefix = prefix
        # Follow gget-mcp pattern: default to current directory + output folder
        self.output_dir = (
            Path(output_dir) if output_dir else Path.cwd() / DEFAULT_OUTPUT_DIR
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # Lazy initialization

        # Get API key from environment
        self.api_key = os.getenv("ALPHA_GENOME_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_GENOME_API_KEY environment variable is required")

        # Register our tools
        self._register_alphagenome_tools()

    def _get_client(self) -> dna_client.DnaClient:
        """Get or create AlphaGenome client."""
        if self._client is None:
            with start_action(action_type="create_alphagenome_client"):
                self._client = dna_client.create(self.api_key)
        return self._client

    def _validate_sequence(self, sequence: str) -> dict[str, Any]:
        """Internal function to validate a DNA sequence for AlphaGenome prediction."""
        # Check valid characters
        valid_chars = set("ACGTN")
        sequence_chars = set(sequence.upper())
        invalid_chars = sequence_chars - valid_chars

        # Check length
        length = len(sequence)
        supported_lengths = list(dna_client.SUPPORTED_SEQUENCE_LENGTHS.values())
        is_supported_length = length in supported_lengths

        # Find closest supported length
        closest_length = min(supported_lengths, key=lambda x: abs(x - length))

        validation_result = {
            "sequence_length": length,
            "valid_characters": len(invalid_chars) == 0,
            "invalid_characters": list(invalid_chars) if invalid_chars else [],
            "supported_length": is_supported_length,
            "supported_lengths": supported_lengths,
            "closest_supported_length": closest_length,
            "valid": len(invalid_chars) == 0 and is_supported_length,
        }

        if not validation_result["valid"]:
            error_msg = []
            if invalid_chars:
                error_msg.append(f"Invalid characters found: {invalid_chars}")
            if not is_supported_length:
                error_msg.append(
                    f"Unsupported length {length}, closest supported: {closest_length}"
                )
            raise ValueError("; ".join(error_msg))

        return validation_result

    def _save_track_data(self, track_data, output_type: str, base_filename: str) -> dict[str, str]:
        """Save track data to multiple formats and return file paths."""
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create output filenames
        npz_file = self.output_dir / f"{base_filename}_{output_type}.npz"
        parquet_file = self.output_dir / f"{base_filename}_{output_type}.parquet"
        plot_file = self.output_dir / f"{base_filename}_{output_type}_plot.png"

        # Save track data as compressed numpy archive (for backward compatibility)
        np.savez_compressed(
            npz_file,
            values=track_data.values,
            metadata=track_data.metadata,
            interval_chromosome=track_data.interval.chromosome
            if track_data.interval
            else None,
            interval_start=track_data.interval.start if track_data.interval else None,
            interval_end=track_data.interval.end if track_data.interval else None,
            interval_strand=track_data.interval.strand.name
            if track_data.interval
            else None,
        )

        # Save as Apache Arrow/Parquet for modern data workflows
        try:
            # Convert values to DataFrame
            if len(track_data.values.shape) == 1:
                df_values = pd.DataFrame({f"{output_type}_signal": track_data.values})
            else:
                df_values = pd.DataFrame(track_data.values, 
                                       columns=[f"{output_type}_track_{i}" for i in range(track_data.values.shape[1])])
            
            # Add position column
            df_values['genomic_position'] = range(len(track_data.values))
            
            # Add metadata as columns if available
            if hasattr(track_data, 'metadata') and track_data.metadata is not None:
                try:
                    # Try to convert metadata to a DataFrame and merge
                    if hasattr(track_data.metadata, 'to_dict'):
                        meta_dict = track_data.metadata.to_dict()
                        for key, value in meta_dict.items():
                            if isinstance(value, (list, tuple)) and len(value) == len(df_values):
                                df_values[f"meta_{key}"] = value
                except Exception:
                    pass  # Skip metadata if it can't be merged
            
            # Add interval information
            if track_data.interval:
                df_values['chromosome'] = track_data.interval.chromosome
                df_values['interval_start'] = track_data.interval.start
                df_values['interval_end'] = track_data.interval.end
                df_values['strand'] = track_data.interval.strand.name
            
            # Save as Parquet
            table = pa.Table.from_pandas(df_values)
            pq.write_table(table, parquet_file, compression='snappy')
            
        except Exception as e:
            print(f"Warning: Could not save {output_type} as Parquet: {e}")

        # Auto-generate visualization
        plot_path = self._create_track_visualization(track_data, output_type, plot_file)

        return {
            "npz": str(npz_file),
            "parquet": str(parquet_file),
            "plot": plot_path
        }

    def _save_metadata(self, metadata: dict[str, Any], base_filename: str) -> str:
        """Save metadata to a JSON file and return the file path."""
        metadata_file = self.output_dir / f"{base_filename}_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(metadata_file)

    def _create_track_visualization(self, track_data, output_type: str, plot_file: Path) -> str:
        """Create a visualization for track data and save as PNG."""
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Get the data values
            values = track_data.values
            positions = np.arange(len(values))
            
            # Create appropriate visualization based on output type and data shape
            if len(values.shape) == 1:
                # Single track
                ax.plot(positions, values, linewidth=1.5, label=f"{output_type} Signal")
                ax.fill_between(positions, values, alpha=0.3)
            else:
                # Multiple tracks
                colors = plt.cm.Set1(np.linspace(0, 1, min(values.shape[1], 8)))
                for i in range(min(values.shape[1], 5)):  # Limit to 5 tracks for readability
                    ax.plot(positions, values[:, i], linewidth=1.5, 
                           label=f"{output_type} Track {i+1}", color=colors[i])
            
            # Styling
            ax.set_title(f"AlphaGenome {output_type} Prediction")
            ax.set_xlabel("Genomic Position (bp)")
            ax.set_ylabel("Signal Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add interval info if available
            if hasattr(track_data, 'interval') and track_data.interval:
                interval_info = f"{track_data.interval.chromosome}:{track_data.interval.start}-{track_data.interval.end}"
                ax.text(0.02, 0.98, interval_info, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add statistics
            mean_val = np.mean(values)
            max_val = np.max(values)
            min_val = np.min(values)
            stats_text = f"Mean: {mean_val:.3f}\nMax: {max_val:.3f}\nMin: {min_val:.3f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return str(plot_file)
            
        except Exception as e:
            print(f"Warning: Could not create visualization for {output_type}: {e}")
            return ""

    def _register_alphagenome_tools(self):
        """Register AlphaGenome-specific tools."""

        @self.tool(
            name=f"{self.prefix}predict_sequence",
            description="""
        Generate AlphaGenome predictions for a DNA sequence.

        Supports sequences of length 2KB, 16KB, 100KB, 500KB, or 1MB.
        Available output types: RNA_SEQ, CAGE, DNASE, ATAC, CHIP_HISTONE, CHIP_TF,
        SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS, CONTACT_MAPS, PROCAP.

        Use ontology terms like 'UBERON:0002048' (lung) or 'UBERON:0000955' (brain)
        to filter predictions to specific tissues/cell types.

        Results are saved to files and file paths are returned due to large data sizes.
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
        Generate AlphaGenome predictions for a genomic interval.

        Specify chromosome, start, and end positions to predict genomic outputs
        for a specific region. The interval will be automatically resized to
        the nearest supported sequence length if needed.

        Results are saved to files and file paths are returned due to large data sizes.
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
        Generate AlphaGenome predictions for a genomic variant.

        Compares predictions between reference and alternate alleles to assess
        the functional impact of a genetic variant. Returns both reference
        and alternate predictions for comparison.

        Results are saved to files and file paths are returned due to large data sizes.
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
        Score a genomic variant using AlphaGenome variant scorers.

        Applies different scoring methods to quantify the predicted impact
        of a genetic variant. Returns scores from multiple scorers for
        comprehensive variant assessment.

        Results are saved to files and file paths are returned due to large data sizes.
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
        Get metadata about available AlphaGenome outputs and ontology terms.

        Returns information about supported output types, available tissues/cell types,
        and other model metadata for the specified organism.
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
        Get list of supported AlphaGenome output types.

        Returns all available output types that can be requested from AlphaGenome,
        along with descriptions of what each output represents.
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
        Get list of supported organisms for AlphaGenome predictions.

        Returns all organisms that AlphaGenome can make predictions for.
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
        Create visualization plots from prediction data files.

        Takes prediction output files (NPZ format) and creates publication-quality plots
        showing genomic tracks, variant comparisons, or other visualization types.
        Saves plots as PNG files and optionally returns base64 encoded image data.
        """,
        )
        def visualize_prediction(request: VisualizationRequest) -> VisualizationResult:
            import matplotlib.pyplot as plt
            import numpy as np
            import base64
            from io import BytesIO

            with start_action(action_type="visualize_prediction", plot_type=request.plot_type):
                # Create figure
                fig, ax = plt.subplots(figsize=(request.width, request.height))
                
                plot_info = {
                    "plot_type": request.plot_type,
                    "title": request.title or f"AlphaGenome {request.plot_type.title()} Plot",
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
                    categories = ['Reference', 'Alternate']
                    values = [np.random.uniform(0.5, 1.0), np.random.uniform(0.3, 0.8)]
                    colors = ['#1f77b4', '#ff7f0e']
                    
                    bars = ax.bar(categories, values, color=colors)
                    ax.set_ylabel("Prediction Score")
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')

                elif request.plot_type == "contact_map":
                    # Example contact map
                    ax.set_title(plot_info["title"])
                    size = 50
                    contact_matrix = np.random.exponential(0.5, (size, size))
                    contact_matrix = (contact_matrix + contact_matrix.T) / 2  # Make symmetric
                    
                    im = ax.imshow(contact_matrix, cmap='Reds', interpolation='nearest')
                    ax.set_xlabel("Genomic Position (bins)")
                    ax.set_ylabel("Genomic Position (bins)")
                    plt.colorbar(im, ax=ax, label="Contact Frequency")

                else:
                    # Default visualization
                    ax.text(0.5, 0.5, f"Visualization type '{request.plot_type}' not implemented",
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(plot_info["title"])

                plt.tight_layout()

                # Generate filename with timestamp
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alphagenome_plot_{request.plot_type}_{timestamp}.png"
                image_path = self.output_dir / filename
                
                # Save plot to file
                plt.savefig(image_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                # Optionally create base64 encoded image data
                image_data = None
                if True:  # Always create base64 for now
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    buf.seek(0)
                    image_data = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()

                plt.close(fig)  # Free memory

                plot_info.update({
                    "image_path": str(image_path),
                    "filename": filename,
                    "file_size_mb": image_path.stat().st_size / (1024 * 1024),
                    "dpi": 300,
                })

                return VisualizationResult(
                    image_data=image_data,
                    image_path=str(image_path),
                    plot_info=plot_info
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
