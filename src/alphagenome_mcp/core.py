#!/usr/bin/env python3
"""Core AlphaGenome functionality shared between MCP server and CLI tools."""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client
from eliot import start_action
from pydantic import BaseModel, Field

matplotlib.use("Agg")  # Use non-interactive backend


class AlphaGenomeConfig(BaseModel):
    """Configuration for AlphaGenome operations."""

    api_key: str = Field(
        description="AlphaGenome API key. Can be set via ALPHA_GENOME_API_KEY environment variable."
    )
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "alphagenome_output",
        description="Directory for output files. Created if it doesn't exist.",
    )
    organism: str = Field(
        default="HOMO_SAPIENS",
        description="Default organism. Options: 'HOMO_SAPIENS' (human), 'MUS_MUSCULUS' (mouse).",
    )
    requested_outputs: list[str] = Field(
        default=["RNA_SEQ", "CAGE", "DNASE", "ATAC"],
        description="Default output types for predictions. Available: RNA_SEQ, CAGE, DNASE, ATAC, "
        "CHIP_HISTONE, CHIP_TF, SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS, "
        "CONTACT_MAPS, PROCAP.",
    )
    ontology_terms: list[str] | None = Field(
        default=None,
        description="Default tissue/cell type ontology terms (UBERON). Examples: "
        "['UBERON:0002048'] for lung, ['UBERON:0000955'] for brain.",
    )
    interval_size: int = Field(
        default=100000,
        description="Default interval size for variant analysis (in bp). Should be one of: "
        "2048, 16384, 98304, 491520, 983040.",
    )

    @classmethod
    def from_env(cls, output_dir: str | Path | None = None) -> "AlphaGenomeConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("ALPHA_GENOME_API_KEY")
        if not api_key:
            raise ValueError("ALPHA_GENOME_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            output_dir=Path(output_dir).resolve()
            if output_dir
            else (Path.cwd() / "alphagenome_output").resolve(),
        )


class VariantInfo(BaseModel):
    """Standardized variant information for processing."""

    chromosome: str = Field(description="Chromosome (e.g., 'chr1', 'chr2', etc.)")
    position: int = Field(description="1-based genomic position")
    reference_bases: str = Field(description="Reference allele sequence")
    alternate_bases: str = Field(description="Alternate allele sequence")
    variant_id: str | None = Field(
        default=None, description="Variant identifier (optional)"
    )

    @property
    def is_snv(self) -> bool:
        """Check if variant is a single nucleotide variant."""
        return (
            len(self.reference_bases) == 1
            and len(self.alternate_bases) == 1
            and self.reference_bases != self.alternate_bases
        )

    def to_alphagenome_variant(self) -> genome.Variant:
        """Convert to AlphaGenome Variant object."""
        return genome.Variant(
            chromosome=self.chromosome,
            position=self.position,
            reference_bases=self.reference_bases,
            alternate_bases=self.alternate_bases,
        )


class AlphaGenomeCore:
    """Core AlphaGenome functionality for predictions and analysis."""

    def __init__(self, config: AlphaGenomeConfig):
        """Initialize with configuration."""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # Lazy initialization

    def get_client(self) -> dna_client.DnaClient:
        """Get or create AlphaGenome client."""
        if self._client is None:
            with start_action(action_type="create_alphagenome_client"):
                self._client = dna_client.create(self.config.api_key)
        return self._client

    def validate_sequence(self, sequence: str) -> dict[str, Any]:
        """Validate a DNA sequence for AlphaGenome prediction."""
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

    def convert_output_types(
        self, output_types: list[str]
    ) -> list[dna_client.OutputType]:
        """Convert string output types to AlphaGenome enum."""
        return [getattr(dna_client.OutputType, ot) for ot in output_types]

    def convert_organism(self, organism: str) -> dna_client.Organism:
        """Convert string organism to AlphaGenome enum."""
        return getattr(dna_client.Organism, organism)

    def create_interval(
        self, chromosome: str, start: int, end: int, strand: str = "POSITIVE"
    ) -> genome.Interval:
        """Create AlphaGenome interval object."""
        strand_enum = getattr(genome.Strand, strand)
        return genome.Interval(
            chromosome=chromosome,
            start=start,
            end=end,
            strand=strand_enum,
        )

    def create_variant_interval(
        self, variant: VariantInfo, interval_size: int | None = None
    ) -> genome.Interval:
        """Create interval around a variant for analysis."""
        if interval_size is None:
            interval_size = self.config.interval_size

        # Center interval around variant
        half_size = interval_size // 2
        start = max(0, variant.position - 1 - half_size)  # Convert to 0-based
        end = start + interval_size

        return self.create_interval(variant.chromosome, start, end)

    def save_track_data(
        self, track_data, output_type: str, base_filename: str
    ) -> dict[str, str]:
        """Save track data to multiple formats and return file paths."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create output filenames
        npz_file = self.config.output_dir / f"{base_filename}_{output_type}.npz"
        parquet_file = self.config.output_dir / f"{base_filename}_{output_type}.parquet"
        plot_file = self.config.output_dir / f"{base_filename}_{output_type}_plot.png"

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
                df_values = pd.DataFrame(
                    track_data.values,
                    columns=[
                        f"{output_type}_track_{i}"
                        for i in range(track_data.values.shape[1])
                    ],
                )

            # Add position column
            df_values["genomic_position"] = range(len(track_data.values))

            # Add metadata as columns if available
            if hasattr(track_data, "metadata") and track_data.metadata is not None:
                try:
                    # Try to convert metadata to a DataFrame and merge
                    if hasattr(track_data.metadata, "to_dict"):
                        meta_dict = track_data.metadata.to_dict()
                        for key, value in meta_dict.items():
                            if isinstance(value, list | tuple) and len(value) == len(
                                df_values
                            ):
                                df_values[f"meta_{key}"] = value
                except Exception:
                    pass  # Skip metadata if it can't be merged

            # Add interval information
            if track_data.interval:
                df_values["chromosome"] = track_data.interval.chromosome
                df_values["interval_start"] = track_data.interval.start
                df_values["interval_end"] = track_data.interval.end
                df_values["strand"] = track_data.interval.strand.name

            # Save as Parquet
            table = pa.Table.from_pandas(df_values)
            pq.write_table(table, parquet_file, compression="snappy")

        except Exception as e:
            print(f"Warning: Could not save {output_type} as Parquet: {e}")

        # Auto-generate visualization
        plot_path = self.create_track_visualization(track_data, output_type, plot_file)

        return {"npz": str(npz_file), "parquet": str(parquet_file), "plot": plot_path}

    def save_metadata(self, metadata: dict[str, Any], base_filename: str) -> str:
        """Save metadata to a JSON file and return the file path."""
        metadata_file = self.config.output_dir / f"{base_filename}_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(metadata_file)

    def create_track_visualization(
        self, track_data, output_type: str, plot_file: Path
    ) -> str:
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
                for i in range(
                    min(values.shape[1], 5)
                ):  # Limit to 5 tracks for readability
                    ax.plot(
                        positions,
                        values[:, i],
                        linewidth=1.5,
                        label=f"{output_type} Track {i + 1}",
                        color=colors[i],
                    )

            # Styling
            ax.set_title(f"AlphaGenome {output_type} Prediction")
            ax.set_xlabel("Genomic Position (bp)")
            ax.set_ylabel("Signal Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add interval info if available
            if hasattr(track_data, "interval") and track_data.interval:
                interval_info = f"{track_data.interval.chromosome}:{track_data.interval.start}-{track_data.interval.end}"
                ax.text(
                    0.02,
                    0.98,
                    interval_info,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
                )

            # Add statistics
            mean_val = np.mean(values)
            max_val = np.max(values)
            min_val = np.min(values)
            stats_text = f"Mean: {mean_val:.3f}\nMax: {max_val:.3f}\nMin: {min_val:.3f}"
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
            )

            plt.tight_layout()

            # Save the plot
            plt.savefig(
                plot_file,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)

            return str(plot_file)

        except Exception as e:
            print(f"Warning: Could not create visualization for {output_type}: {e}")
            return ""

    def predict_sequence(
        self,
        sequence: str,
        requested_outputs: list[str] | None = None,
        organism: str | None = None,
        ontology_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Predict genomic outputs for a DNA sequence."""
        with start_action(
            action_type="predict_sequence", sequence_length=len(sequence)
        ):
            # Use defaults from config if not provided
            requested_outputs = requested_outputs or self.config.requested_outputs
            organism = organism or self.config.organism
            ontology_terms = ontology_terms or self.config.ontology_terms

            # Validate sequence
            self.validate_sequence(sequence)

            client = self.get_client()

            # Convert parameters
            output_types = self.convert_output_types(requested_outputs)
            organism_enum = self.convert_organism(organism)

            # Make prediction
            output = client.predict_sequence(
                sequence=sequence,
                organism=organism_enum,
                requested_outputs=output_types,
                ontology_terms=ontology_terms,
            )

            # Generate base filename
            base_filename = (
                f"sequence_prediction_{len(sequence)}bp_{hash(sequence) % 100000}"
            )

            # Save outputs to files
            output_files = {}
            for output_type in requested_outputs:
                attr_name = output_type.lower()
                if hasattr(output, attr_name):
                    track_data = getattr(output, attr_name)
                    if track_data is not None:
                        file_paths = self.save_track_data(
                            track_data, output_type, base_filename
                        )
                        output_files[output_type] = file_paths

            # Save metadata
            metadata = {
                "sequence_length": len(sequence),
                "organism": organism,
                "output_types": requested_outputs,
                "ontology_terms": ontology_terms,
                "sequence_hash": hash(sequence) % 100000,
            }
            metadata_file = self.save_metadata(metadata, base_filename)

            return {
                "output_files": output_files,
                "metadata_file": metadata_file,
                "base_filename": base_filename,
            }

    def predict_variant(
        self,
        variant: VariantInfo,
        requested_outputs: list[str] | None = None,
        organism: str | None = None,
        ontology_terms: list[str] | None = None,
        interval_size: int | None = None,
    ) -> dict[str, Any]:
        """Predict variant impact by comparing reference vs alternate."""
        with start_action(
            action_type="predict_variant",
            chromosome=variant.chromosome,
            variant_position=variant.position,
        ):
            # Use defaults from config if not provided
            requested_outputs = requested_outputs or self.config.requested_outputs
            organism = organism or self.config.organism
            ontology_terms = ontology_terms or self.config.ontology_terms
            interval_size = interval_size or self.config.interval_size

            client = self.get_client()

            # Create interval and variant objects
            interval = self.create_variant_interval(variant, interval_size)
            ag_variant = variant.to_alphagenome_variant()

            # Convert parameters
            output_types = self.convert_output_types(requested_outputs)
            organism_enum = self.convert_organism(organism)

            # Make prediction
            variant_output = client.predict_variant(
                interval=interval,
                variant=ag_variant,
                organism=organism_enum,
                requested_outputs=output_types,
                ontology_terms=ontology_terms,
            )

            # Generate base filename
            base_filename = f"variant_prediction_{variant.chromosome}_{variant.position}_{variant.reference_bases}_{variant.alternate_bases}"

            # Save reference and alternate outputs
            def save_output_data(output_obj, suffix):
                output_files = {}
                for output_type in requested_outputs:
                    attr_name = output_type.lower()
                    if hasattr(output_obj, attr_name):
                        track_data = getattr(output_obj, attr_name)
                        if track_data is not None:
                            file_paths = self.save_track_data(
                                track_data, output_type, f"{base_filename}_{suffix}"
                            )
                            output_files[output_type] = file_paths
                return output_files

            reference_files = save_output_data(variant_output.reference, "reference")
            alternate_files = save_output_data(variant_output.alternate, "alternate")

            # Save combined file lists
            reference_file = (
                self.config.output_dir / f"{base_filename}_reference_files.json"
            )
            alternate_file = (
                self.config.output_dir / f"{base_filename}_alternate_files.json"
            )

            with open(reference_file, "w") as f:
                json.dump(reference_files, f, indent=2)
            with open(alternate_file, "w") as f:
                json.dump(alternate_files, f, indent=2)

            # Save metadata
            metadata = {
                "organism": organism,
                "output_types": requested_outputs,
                "ontology_terms": ontology_terms,
                "variant": variant.model_dump(),
                "interval": {
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "width": interval.width,
                },
            }
            metadata_file = self.save_metadata(metadata, base_filename)

            return {
                "reference_file": str(reference_file),
                "alternate_file": str(alternate_file),
                "metadata_file": metadata_file,
                "variant": variant.model_dump(),
                "interval": {
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "width": interval.width,
                },
                "base_filename": base_filename,
            }

    def score_variant(
        self,
        variant: VariantInfo,
        variant_scorers: list[str] | None = None,
        organism: str | None = None,
        interval_size: int | None = None,
    ) -> dict[str, Any]:
        """Score variant for pathogenicity and functional impact."""
        with start_action(
            action_type="score_variant",
            chromosome=variant.chromosome,
            variant_position=variant.position,
        ):
            # Use defaults from config if not provided
            organism = organism or self.config.organism
            interval_size = interval_size or self.config.interval_size

            client = self.get_client()

            # Create interval and variant objects
            interval = self.create_variant_interval(variant, interval_size)
            ag_variant = variant.to_alphagenome_variant()

            # Get organism and scorers
            organism_enum = self.convert_organism(organism)

            if variant_scorers:
                # Convert scorer names to objects
                raise NotImplementedError("Custom scorer selection not yet implemented")
            else:
                # Use recommended scorers
                scorer_objects = None

            # Score variant
            scores = client.score_variant(
                interval=interval,
                variant=ag_variant,
                variant_scorers=scorer_objects,
                organism=organism_enum,
            )

            # Generate base filename
            base_filename = f"variant_scoring_{variant.chromosome}_{variant.position}_{variant.reference_bases}_{variant.alternate_bases}"

            # Save AnnData objects
            score_files = []
            for i, score_data in enumerate(scores):
                score_file = self.config.output_dir / f"{base_filename}_scorer_{i}.h5ad"
                score_data.write_h5ad(score_file)
                score_files.append(str(score_file))

            # Create scores file list
            scores_file = self.config.output_dir / f"{base_filename}_scores.json"
            with open(scores_file, "w") as f:
                json.dump(
                    {"score_files": score_files, "scorer_count": len(score_files)},
                    f,
                    indent=2,
                )

            # Save metadata
            metadata = {
                "organism": organism,
                "variant_scorers": variant_scorers,
                "variant": variant.model_dump(),
                "interval": {
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "width": interval.width,
                },
                "scorer_count": len(scores),
            }
            metadata_file = self.save_metadata(metadata, base_filename)

            return {
                "scores_file": str(scores_file),
                "metadata_file": metadata_file,
                "variant": variant.model_dump(),
                "interval": {
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "width": interval.width,
                },
                "base_filename": base_filename,
            }
