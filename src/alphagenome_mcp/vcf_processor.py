#!/usr/bin/env python3
"""VCF processing functionality for AlphaGenome variant annotation."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from eliot import start_action
from pydantic import BaseModel, Field

from alphagenome_mcp.core import AlphaGenomeConfig, AlphaGenomeCore, VariantInfo


class VCFVariant(VariantInfo):
    """Variant information from VCF with additional metadata."""

    quality: float | None = Field(
        default=None, description="Variant quality score from VCF"
    )
    filter_status: str | None = Field(
        default=None, description="Filter status from VCF"
    )
    info: dict[str, Any] = Field(
        default_factory=dict, description="INFO field from VCF"
    )
    format_data: dict[str, Any] = Field(
        default_factory=dict, description="FORMAT data from VCF"
    )
    samples: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Sample data from VCF"
    )

    @classmethod
    def from_vcf_record(cls, record, variant_id: str | None = None) -> "VCFVariant":
        """Create VCFVariant from a VCF record (assuming cyvcf2 or similar)."""
        return cls(
            chromosome=record.CHROM,
            position=record.POS,
            reference_bases=record.REF,
            alternate_bases=record.ALT[0]
            if record.ALT
            else "",  # Take first ALT for now
            variant_id=variant_id or getattr(record, "ID", None),
            quality=getattr(record, "QUAL", None),
            filter_status=getattr(record, "FILTER", None),
            info=dict(record.INFO) if hasattr(record, "INFO") else {},
        )


class VCFProcessingConfig(BaseModel):
    """Configuration for VCF processing operations."""

    alphagenome_config: AlphaGenomeConfig = Field(
        description="Core AlphaGenome configuration"
    )
    batch_size: int = Field(
        default=50, description="Number of variants to process in parallel"
    )
    max_variants: int | None = Field(
        default=None, description="Maximum number of variants to process (None for all)"
    )
    include_predictions: bool = Field(
        default=True, description="Include detailed predictions"
    )
    include_scoring: bool = Field(default=True, description="Include variant scoring")
    output_format: str = Field(
        default="vcf", description="Output format: 'vcf', 'tsv', 'json'"
    )
    annotation_prefix: str = Field(
        default="AG_", description="Prefix for annotation fields"
    )

    @classmethod
    def from_args(
        cls,
        vcf_file: Path,
        output_dir: Path | None = None,
        api_key: str | None = None,
        organism: str = "HOMO_SAPIENS",
        outputs: list[str] | None = None,
        batch_size: int = 50,
        max_variants: int | None = None,
        **kwargs,
    ) -> "VCFProcessingConfig":
        """Create configuration from command line arguments."""

        # Create AlphaGenome config
        ag_config = AlphaGenomeConfig(
            api_key=api_key or AlphaGenomeConfig.from_env().api_key,
            output_dir=output_dir or Path.cwd() / "alphagenome_vcf_output",
            organism=organism,
            requested_outputs=outputs or ["RNA_SEQ", "CAGE", "DNASE", "ATAC"],
        )

        return cls(
            alphagenome_config=ag_config,
            batch_size=batch_size,
            max_variants=max_variants,
            **kwargs,
        )


class VCFProcessor:
    """Process VCF files with AlphaGenome annotations."""

    def __init__(self, config: VCFProcessingConfig):
        """Initialize VCF processor with configuration."""
        self.config = config
        self.core = AlphaGenomeCore(config.alphagenome_config)
        self.processed_variants = []
        self.failed_variants = []

    def read_vcf_variants(self, vcf_file: Path) -> Iterator[VCFVariant]:
        """Read variants from VCF file."""
        try:
            import cyvcf2

            vcf = cyvcf2.VCF(str(vcf_file))
            count = 0

            for record in vcf:
                if self.config.max_variants and count >= self.config.max_variants:
                    break

                # Handle multi-allelic variants by processing each ALT separately
                for i, alt in enumerate(record.ALT):
                    variant_id = (
                        f"{record.ID}_{i}"
                        if record.ID and len(record.ALT) > 1
                        else record.ID
                    )

                    # Create variant with specific ALT allele
                    variant = VCFVariant(
                        chromosome=record.CHROM,
                        position=record.POS,
                        reference_bases=record.REF,
                        alternate_bases=alt,
                        variant_id=variant_id,
                        quality=record.QUAL,
                        filter_status=record.FILTER,
                        info=dict(record.INFO) if hasattr(record, "INFO") else {},
                    )

                    yield variant
                    count += 1

                    if self.config.max_variants and count >= self.config.max_variants:
                        break

        except ImportError:
            # Fallback to basic pandas reading for simple VCF files
            typer.echo("cyvcf2 not available, using basic VCF parsing", err=True)

            # Read VCF with pandas (skipping header lines)
            with open(vcf_file) as f:
                lines = [line for line in f if not line.startswith("##")]

            if not lines:
                return

            # Find header line
            header_line = next(
                (line for line in lines if line.startswith("#CHROM")), None
            )
            if not header_line:
                raise ValueError("No VCF header line found")

            # Read data
            header_index = lines.index(header_line)
            df = pd.read_csv(
                vcf_file,
                sep="\t",
                skiprows=header_index,
                nrows=self.config.max_variants,
            )

            for _, row in df.iterrows():
                variant = VCFVariant(
                    chromosome=row["#CHROM"],
                    position=int(row["POS"]),
                    reference_bases=row["REF"],
                    alternate_bases=row["ALT"],
                    variant_id=row.get("ID"),
                    quality=row.get("QUAL"),
                    filter_status=row.get("FILTER"),
                )
                yield variant

    def process_variant(self, variant: VCFVariant) -> dict[str, Any]:
        """Process a single variant with AlphaGenome."""
        with start_action(
            action_type="process_vcf_variant", variant_id=variant.variant_id
        ):
            results = {
                "variant": variant.model_dump(),
                "success": True,
                "error": None,
            }

            try:
                # Get predictions if requested
                if self.config.include_predictions:
                    predictions = self.core.predict_variant(variant)
                    results["predictions"] = predictions

                # Get scores if requested
                if self.config.include_scoring:
                    scores = self.core.score_variant(variant)
                    results["scores"] = scores

                return results

            except Exception as e:
                results["success"] = False
                results["error"] = str(e)
                typer.echo(
                    f"Error processing variant {variant.variant_id}: {e}", err=True
                )
                return results

    def process_variants_batch(
        self, variants: list[VCFVariant]
    ) -> list[dict[str, Any]]:
        """Process a batch of variants."""
        results = []

        for variant in variants:
            result = self.process_variant(variant)
            results.append(result)

            if result["success"]:
                self.processed_variants.append(variant)
            else:
                self.failed_variants.append(variant)

        return results

    def process_vcf_file(self, vcf_file: Path) -> dict[str, Any]:
        """Process entire VCF file."""
        typer.echo(f"Processing VCF file: {vcf_file}")
        typer.echo(f"Output directory: {self.config.alphagenome_config.output_dir}")

        # Create output directory
        self.config.alphagenome_config.output_dir.mkdir(parents=True, exist_ok=True)

        # Track processing stats
        total_variants = 0
        processed_count = 0
        failed_count = 0
        all_results = []

        # Process variants in batches
        batch = []

        with start_action(action_type="process_vcf_file", vcf_file=str(vcf_file)):
            for variant in self.read_vcf_variants(vcf_file):
                batch.append(variant)
                total_variants += 1

                # Process batch when full
                if len(batch) >= self.config.batch_size:
                    batch_results = self.process_variants_batch(batch)
                    all_results.extend(batch_results)

                    # Update counters
                    processed_count += sum(1 for r in batch_results if r["success"])
                    failed_count += sum(1 for r in batch_results if not r["success"])

                    # Progress update
                    typer.echo(
                        f"Processed {processed_count}/{total_variants} variants "
                        f"({failed_count} failed)"
                    )

                    batch = []

            # Process remaining variants
            if batch:
                batch_results = self.process_variants_batch(batch)
                all_results.extend(batch_results)
                processed_count += sum(1 for r in batch_results if r["success"])
                failed_count += sum(1 for r in batch_results if not r["success"])

        # Save summary results
        summary = {
            "vcf_file": str(vcf_file),
            "total_variants": total_variants,
            "processed_variants": processed_count,
            "failed_variants": failed_count,
            "output_dir": str(self.config.alphagenome_config.output_dir),
            "config": self.config.model_dump(),
        }

        # Save detailed results
        results_file = (
            self.config.alphagenome_config.output_dir / "vcf_processing_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": all_results,
                },
                f,
                indent=2,
                default=str,
            )

        # Save summary
        summary_file = (
            self.config.alphagenome_config.output_dir / "processing_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        typer.echo("\nProcessing complete!")
        typer.echo(f"Total variants: {total_variants}")
        typer.echo(f"Successfully processed: {processed_count}")
        typer.echo(f"Failed: {failed_count}")
        typer.echo(f"Results saved to: {results_file}")

        return summary

    def create_annotated_vcf(
        self, vcf_file: Path, results: list[dict[str, Any]]
    ) -> Path:
        """Create annotated VCF file with AlphaGenome predictions."""
        output_file = (
            self.config.alphagenome_config.output_dir / f"{vcf_file.stem}_annotated.vcf"
        )

        # This would implement VCF writing with added INFO fields
        # For now, create a TSV summary
        output_file = output_file.with_suffix(".tsv")

        rows = []
        for result in results:
            if not result["success"]:
                continue

            variant = result["variant"]
            row = {
                "chromosome": variant["chromosome"],
                "position": variant["position"],
                "variant_id": variant.get("variant_id", "."),
                "reference": variant["reference_bases"],
                "alternate": variant["alternate_bases"],
                "is_snv": variant.get("is_snv", False),
            }

            # Add prediction file paths
            if "predictions" in result:
                pred = result["predictions"]
                row["prediction_metadata"] = pred.get("metadata_file")
                row["reference_files"] = pred.get("reference_file")
                row["alternate_files"] = pred.get("alternate_file")

            # Add scoring file paths
            if "scores" in result:
                score = result["scores"]
                row["scores_file"] = score.get("scores_file")
                row["scoring_metadata"] = score.get("metadata_file")

            rows.append(row)

        # Save as TSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, sep="\t", index=False)

        typer.echo(f"Annotated results saved to: {output_file}")
        return output_file


# CLI helper functions for the VCF processor
def create_vcf_processor_from_args(
    vcf_file: Path,
    output_dir: Path | None = None,
    api_key: str | None = None,
    organism: str = "HOMO_SAPIENS",
    outputs: list[str] | None = None,
    batch_size: int = 50,
    max_variants: int | None = None,
    include_predictions: bool = True,
    include_scoring: bool = True,
) -> VCFProcessor:
    """Create VCF processor from command line arguments."""
    config = VCFProcessingConfig.from_args(
        vcf_file=vcf_file,
        output_dir=output_dir,
        api_key=api_key,
        organism=organism,
        outputs=outputs,
        batch_size=batch_size,
        max_variants=max_variants,
        include_predictions=include_predictions,
        include_scoring=include_scoring,
    )
    return VCFProcessor(config)
