#!/usr/bin/env python3
"""Example of using the VCF processor for batch variant annotation."""

from pathlib import Path

from alphagenome_mcp.core import AlphaGenomeConfig, VariantInfo
from alphagenome_mcp.vcf_processor import VCFProcessor, VCFProcessingConfig, VCFVariant


def create_test_vcf():
    """Create a simple test VCF file for demonstration."""
    vcf_content = """##fileformat=VCFv4.2
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	123456	rs123456	A	T	60	PASS	DP=30	GT	0/1
chr1	234567	rs234567	G	C	45	PASS	DP=25	GT	1/1
chr2	345678	.	ATG	A	70	PASS	DP=35	GT	0/1
chr3	456789	rs456789	C	G	55	PASS	DP=28	GT	0/1
chrX	567890	.	T	TA	65	PASS	DP=32	GT	0/1
"""
    
    test_vcf = Path("test_variants.vcf")
    with open(test_vcf, "w") as f:
        f.write(vcf_content)
    
    print(f"Created test VCF file: {test_vcf}")
    return test_vcf


def main():
    """Demonstrate VCF processing functionality."""
    
    print("=== VCF Processor Example ===\n")
    
    # Create a test VCF file
    vcf_file = create_test_vcf()
    
    try:
        # Example 1: Basic VCF processing configuration
        print("=== Basic Configuration ===")
        
        # Create configuration for VCF processing
        # Note: This would normally use a real API key from environment
        try:
            ag_config = AlphaGenomeConfig.from_env(
                output_dir=Path("vcf_output_example")
            )
        except ValueError:
            # Fallback for demo purposes
            ag_config = AlphaGenomeConfig(
                api_key="demo_key",  # Would be real API key
                output_dir=Path("vcf_output_example"),
                organism="HOMO_SAPIENS",
                requested_outputs=["RNA_SEQ", "DNASE", "ATAC"],
            )
        
        vcf_config = VCFProcessingConfig(
            alphagenome_config=ag_config,
            batch_size=2,  # Small batch for demo
            max_variants=3,  # Only process first 3 variants
            include_predictions=True,
            include_scoring=True,
        )
        
        print(f"VCF Processing Configuration:")
        print(f"  Input file: {vcf_file}")
        print(f"  Output directory: {vcf_config.alphagenome_config.output_dir}")
        print(f"  Batch size: {vcf_config.batch_size}")
        print(f"  Max variants: {vcf_config.max_variants}")
        print(f"  Include predictions: {vcf_config.include_predictions}")
        print(f"  Include scoring: {vcf_config.include_scoring}")
        print()
        
        # Example 2: Reading VCF variants
        print("=== Reading VCF Variants ===")
        
        processor = VCFProcessor(vcf_config)
        
        variants = list(processor.read_vcf_variants(vcf_file))
        print(f"Found {len(variants)} variants:")
        
        for i, variant in enumerate(variants):
            print(f"  {i+1}. {variant.variant_id or 'unnamed'}")
            print(f"     {variant.chromosome}:{variant.position} {variant.reference_bases}→{variant.alternate_bases}")
            print(f"     Is SNV: {variant.is_snv}")
            print(f"     Quality: {variant.quality}")
            print()
        
        # Example 3: Convert to core variant format
        print("=== Converting to Core Format ===")
        
        for variant in variants:
            # VCFVariant inherits from VariantInfo, so it works directly
            ag_variant = variant.to_alphagenome_variant()
            print(f"Variant {variant.variant_id}: {ag_variant.chromosome}:{ag_variant.position}")
        print()
        
        # Example 4: Simulated processing (without actual API calls)
        print("=== Simulated Processing ===")
        
        # Note: This would normally call the actual AlphaGenome API
        # For demo, we'll just simulate the structure
        
        simulated_results = []
        for variant in variants:
            result = {
                "variant": variant.model_dump(),
                "success": True,
                "error": None,
                # These would be real file paths from actual processing:
                "predictions": {
                    "reference_file": f"ref_{variant.chromosome}_{variant.position}.json",
                    "alternate_file": f"alt_{variant.chromosome}_{variant.position}.json",
                    "metadata_file": f"meta_{variant.chromosome}_{variant.position}.json",
                },
                "scores": {
                    "scores_file": f"scores_{variant.chromosome}_{variant.position}.json",
                    "metadata_file": f"score_meta_{variant.chromosome}_{variant.position}.json",
                }
            }
            simulated_results.append(result)
        
        print(f"Simulated processing of {len(simulated_results)} variants:")
        for i, result in enumerate(simulated_results):
            variant_info = result["variant"]
            print(f"  {i+1}. {variant_info['variant_id']} - {'✓ Success' if result['success'] else '❌ Failed'}")
        print()
        
        # Example 5: Configuration from command line style arguments
        print("=== CLI-style Configuration ===")
        
        from alphagenome_mcp.vcf_processor import create_vcf_processor_from_args
        
        # This shows how you'd create a processor from CLI arguments
        cli_processor = create_vcf_processor_from_args(
            vcf_file=vcf_file,
            output_dir=Path("cli_output"),
            organism="HOMO_SAPIENS",
            outputs=["RNA_SEQ", "CAGE"],
            batch_size=5,
            max_variants=10,
            include_predictions=True,
            include_scoring=False,  # Skip scoring for faster processing
        )
        
        print(f"CLI-style processor created:")
        print(f"  Batch size: {cli_processor.config.batch_size}")
        print(f"  Outputs: {cli_processor.config.alphagenome_config.requested_outputs}")
        print(f"  Include scoring: {cli_processor.config.include_scoring}")
        
        # Example 6: Show how this integrates with the CLI
        print("\n=== CLI Integration ===")
        print("To use this with the actual CLI:")
        print(f"  python -m alphagenome_mcp.cli annotate-vcf {vcf_file}")
        print(f"  python -m alphagenome_mcp.cli annotate-vcf {vcf_file} -t RNA_SEQ -t DNASE")
        print(f"  python -m alphagenome_mcp.cli annotate-vcf {vcf_file} --max-variants 5")
        print(f"  python -m alphagenome_mcp.cli test-variant chr1 123456 A T")
        
    finally:
        # Clean up test file
        if vcf_file.exists():
            vcf_file.unlink()
            print(f"\nCleaned up test file: {vcf_file}")
    
    print("\n=== VCF Example Complete ===")
    print("This demonstrates how to use the VCF processor for batch")
    print("variant annotation with AlphaGenome predictions.")


if __name__ == "__main__":
    main()