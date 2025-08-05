#!/usr/bin/env python3
"""AlphaGenome CLI for VCF annotation and variant analysis."""

from pathlib import Path
from typing import Annotated

import typer
from eliot import start_action

from alphagenome_mcp.core import AlphaGenomeConfig
from alphagenome_mcp.vcf_processor import (
    create_vcf_processor_from_args,
)

app = typer.Typer(
    help="AlphaGenome CLI - Annotate VCF files with AlphaGenome genomic predictions",
    invoke_without_command=True,
)


@app.callback()
def main(ctx: typer.Context) -> None:
    """
    AlphaGenome CLI - Annotate VCF files with AlphaGenome genomic predictions.
    
    If no command is specified, help will be shown.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("annotate-vcf")
def annotate_vcf(
    vcf_file: Annotated[Path, typer.Argument(help="Input VCF file to annotate")],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for results (default: ./alphagenome_vcf_output)",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="AlphaGenome API key (or use ALPHA_GENOME_API_KEY env var)",
        ),
    ] = None,
    organism: Annotated[
        str, typer.Option("--organism", "-g", help="Target organism")
    ] = "HOMO_SAPIENS",
    outputs: Annotated[
        list[str] | None,
        typer.Option(
            "--output",
            "-t",
            help="Output types to predict (can be specified multiple times)",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", "-b", help="Number of variants to process in parallel"
        ),
    ] = 50,
    max_variants: Annotated[
        int | None,
        typer.Option(
            "--max-variants",
            "-m",
            help="Maximum number of variants to process (for testing)",
        ),
    ] = None,
    include_predictions: Annotated[
        bool,
        typer.Option(
            "--predictions/--no-predictions",
            help="Include detailed variant predictions",
        ),
    ] = True,
    include_scoring: Annotated[
        bool,
        typer.Option(
            "--scoring/--no-scoring", help="Include variant pathogenicity scoring"
        ),
    ] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """
    Annotate a VCF file with AlphaGenome predictions and scores.

    This command processes variants in a VCF file and adds AlphaGenome
    predictions for functional impact assessment.

    EXAMPLES:

    Basic annotation:
    alphagenome-cli annotate-vcf variants.vcf

    Custom output types:
    alphagenome-cli annotate-vcf variants.vcf -t RNA_SEQ -t DNASE -t ATAC

    Process subset for testing:
    alphagenome-cli annotate-vcf variants.vcf --max-variants 10

    Scoring only (faster):
    alphagenome-cli annotate-vcf variants.vcf --no-predictions --scoring
    """

    # Validate VCF file exists
    if not vcf_file.exists():
        typer.echo(f"Error: VCF file not found: {vcf_file}", err=True)
        raise typer.Exit(1)

    # Set default outputs if none provided
    if outputs is None:
        outputs = ["RNA_SEQ", "CAGE", "DNASE", "ATAC"]

    with start_action(action_type="annotate_vcf_cli", vcf_file=str(vcf_file)):
        try:
            # Create processor
            processor = create_vcf_processor_from_args(
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

            if verbose:
                typer.echo("Configuration:")
                typer.echo(f"  VCF file: {vcf_file}")
                typer.echo(
                    f"  Output dir: {processor.config.alphagenome_config.output_dir}"
                )
                typer.echo(f"  Organism: {organism}")
                typer.echo(f"  Output types: {outputs}")
                typer.echo(f"  Batch size: {batch_size}")
                typer.echo(f"  Max variants: {max_variants or 'all'}")
                typer.echo(f"  Include predictions: {include_predictions}")
                typer.echo(f"  Include scoring: {include_scoring}")
                typer.echo()

            # Process VCF file
            summary = processor.process_vcf_file(vcf_file)

            # Create annotated output
            # Note: In a full implementation, this would read the results
            # and create an actual annotated VCF file
            typer.echo("\nAnnotation complete! Summary:")
            typer.echo(f"  Total variants: {summary['total_variants']}")
            typer.echo(f"  Successfully processed: {summary['processed_variants']}")
            typer.echo(f"  Failed: {summary['failed_variants']}")
            typer.echo(f"  Results directory: {summary['output_dir']}")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            if verbose:
                import traceback

                traceback.print_exc()
            raise typer.Exit(1)


@app.command("test-variant")
def test_variant(
    chromosome: Annotated[str, typer.Argument(help="Chromosome (e.g., chr1)")],
    position: Annotated[int, typer.Argument(help="Genomic position (1-based)")],
    ref: Annotated[str, typer.Argument(help="Reference allele")],
    alt: Annotated[str, typer.Argument(help="Alternate allele")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Output directory for results"),
    ] = None,
    api_key: Annotated[
        str | None, typer.Option("--api-key", "-k", help="AlphaGenome API key")
    ] = None,
    organism: Annotated[str, typer.Option("--organism", "-g")] = "HOMO_SAPIENS",
    outputs: Annotated[list[str] | None, typer.Option("--output", "-t")] = None,
    include_predictions: Annotated[
        bool, typer.Option("--predictions/--no-predictions")
    ] = True,
    include_scoring: Annotated[bool, typer.Option("--scoring/--no-scoring")] = True,
) -> None:
    """
    Test AlphaGenome analysis on a single variant.

    EXAMPLES:

    Test a SNV:
    alphagenome-cli test-variant chr1 123456 A T

    Test with specific outputs:
    alphagenome-cli test-variant chr1 123456 A T -t RNA_SEQ -t DNASE
    """

    from alphagenome_mcp.core import VariantInfo

    # Set defaults
    if outputs is None:
        outputs = ["RNA_SEQ", "CAGE", "DNASE", "ATAC"]

    with start_action(action_type="test_variant_cli"):
        try:
            # Create configuration
            config = AlphaGenomeConfig.from_env(output_dir=output_dir)
            if api_key:
                config.api_key = api_key
            config.organism = organism
            config.requested_outputs = outputs

            # Import here to avoid circular imports
            from alphagenome_mcp.core import AlphaGenomeCore

            core = AlphaGenomeCore(config)

            # Create variant
            variant = VariantInfo(
                chromosome=chromosome,
                position=position,
                reference_bases=ref,
                alternate_bases=alt,
                variant_id=f"{chromosome}_{position}_{ref}_{alt}",
            )

            typer.echo(f"Testing variant: {variant.variant_id}")
            typer.echo(f"Is SNV: {variant.is_snv}")
            typer.echo(f"Output directory: {config.output_dir}")

            results = {}

            # Get predictions if requested
            if include_predictions:
                typer.echo("\nRunning variant predictions...")
                predictions = core.predict_variant(variant)
                results["predictions"] = predictions
                typer.echo(f"Predictions saved to: {predictions['metadata_file']}")

            # Get scores if requested
            if include_scoring:
                typer.echo("\nRunning variant scoring...")
                scores = core.score_variant(variant)
                results["scores"] = scores
                typer.echo(f"Scores saved to: {scores['metadata_file']}")

            typer.echo(f"\nAnalysis complete! Results in: {config.output_dir}")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)


@app.command("list-outputs")
def list_outputs() -> None:
    """List available AlphaGenome output types."""

    outputs = {
        "RNA_SEQ": "RNA sequencing gene expression tracks",
        "CAGE": "Cap Analysis Gene Expression (transcription start sites)",
        "DNASE": "DNase I hypersensitive sites (chromatin accessibility)",
        "ATAC": "ATAC-seq chromatin accessibility",
        "CHIP_HISTONE": "ChIP-seq histone modification tracks",
        "CHIP_TF": "ChIP-seq transcription factor binding sites",
        "SPLICE_SITES": "Splice donor and acceptor site predictions",
        "SPLICE_SITE_USAGE": "Quantitative splice site usage",
        "SPLICE_JUNCTIONS": "RNA-seq splice junction tracks",
        "CONTACT_MAPS": "3D chromatin interaction contact maps",
        "PROCAP": "Precision Run-On sequencing and capping",
    }

    typer.echo("Available AlphaGenome output types:")
    typer.echo()

    for output_type, description in outputs.items():
        typer.echo(f"  {output_type:<18} {description}")

    typer.echo()
    typer.echo("Usage: Use with -t/--output flag, e.g.:")
    typer.echo("  alphagenome-cli annotate-vcf file.vcf -t RNA_SEQ -t DNASE")


@app.command("check-config")
def check_config(
    api_key: Annotated[
        str | None, typer.Option("--api-key", "-k", help="AlphaGenome API key to test")
    ] = None,
) -> None:
    """Check AlphaGenome configuration and connectivity."""

    try:
        # Test configuration
        if api_key:
            config = AlphaGenomeConfig(
                api_key=api_key, output_dir=Path.cwd() / "test_output"
            )
        else:
            config = AlphaGenomeConfig.from_env()

        typer.echo("✓ Configuration loaded successfully")
        typer.echo(f"  Output directory: {config.output_dir}")
        typer.echo(f"  Default organism: {config.organism}")
        typer.echo(f"  Default outputs: {', '.join(config.requested_outputs)}")

        # Test client creation
        from alphagenome_mcp.core import AlphaGenomeCore

        core = AlphaGenomeCore(config)

        typer.echo("\nTesting AlphaGenome client connection...")
        core.get_client()
        typer.echo("✓ Successfully connected to AlphaGenome API")

        # Test a small validation
        test_sequence = "ATCGATCGATCG" * 170  # ~2KB sequence
        validation = core.validate_sequence(test_sequence)
        typer.echo(
            f"✓ Sequence validation working (test sequence: {validation['sequence_length']}bp)"
        )

        typer.echo("\n🎉 Configuration and connectivity check passed!")

    except Exception as e:
        typer.echo(f"❌ Configuration check failed: {e}", err=True)
        typer.echo("\nTroubleshooting:")
        typer.echo("  1. Set ALPHA_GENOME_API_KEY environment variable")
        typer.echo("  2. Or use --api-key flag")
        typer.echo("  3. Ensure you have valid AlphaGenome API access")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
