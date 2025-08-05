#!/usr/bin/env python3
"""Example of using the shared AlphaGenome core functionality."""

from pathlib import Path

from alphagenome_mcp.core import AlphaGenomeConfig, AlphaGenomeCore, VariantInfo


def main():
    """Demonstrate using the core AlphaGenome functionality."""
    
    # Example 1: Basic configuration setup
    print("=== AlphaGenome Core Usage Example ===\n")
    
    # Create configuration (will use environment variable for API key)
    try:
        config = AlphaGenomeConfig.from_env(
            output_dir=Path("example_output")
        )
        print(f"✓ Configuration created:")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Default organism: {config.organism}")
        print(f"  Default outputs: {config.requested_outputs}")
        print()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set the ALPHA_GENOME_API_KEY environment variable")
        return
    
    # Create core instance
    core = AlphaGenomeCore(config)
    
    # Example 2: Test sequence validation
    print("=== Sequence Validation ===")
    test_sequences = [
        "ATCGATCG" * 256,  # 2KB sequence
        "ATCGATCG" * 2048,  # 16KB sequence  
        "ATCG",  # Too short
        "ATCGXYZ",  # Invalid characters
    ]
    
    for i, seq in enumerate(test_sequences):
        try:
            validation = core.validate_sequence(seq)
            print(f"Sequence {i+1}: ✓ Valid ({validation['sequence_length']}bp)")
        except ValueError as e:
            print(f"Sequence {i+1}: ❌ Invalid - {e}")
    print()
    
    # Example 3: Single variant analysis
    print("=== Single Variant Analysis ===")
    
    # Create a test variant
    variant = VariantInfo(
        chromosome="chr1",
        position=123456,
        reference_bases="A",
        alternate_bases="T",
        variant_id="rs123456"
    )
    
    print(f"Analyzing variant: {variant.variant_id}")
    print(f"  Location: {variant.chromosome}:{variant.position}")
    print(f"  Change: {variant.reference_bases} → {variant.alternate_bases}")
    print(f"  Is SNV: {variant.is_snv}")
    
    # Note: Actual prediction calls would require valid API access
    # Uncomment the following lines if you have API access:
    
    # try:
    #     # Get variant predictions
    #     print("\nRunning variant predictions...")
    #     predictions = core.predict_variant(
    #         variant=variant,
    #         requested_outputs=["RNA_SEQ", "DNASE"],
    #     )
    #     print(f"✓ Predictions completed:")
    #     print(f"  Reference files: {predictions['reference_file']}")
    #     print(f"  Alternate files: {predictions['alternate_file']}")
    #     print(f"  Metadata: {predictions['metadata_file']}")
    #     
    #     # Get variant scores
    #     print("\nRunning variant scoring...")
    #     scores = core.score_variant(variant=variant)
    #     print(f"✓ Scoring completed:")
    #     print(f"  Scores file: {scores['scores_file']}")
    #     print(f"  Metadata: {scores['metadata_file']}")
    #     
    # except Exception as e:
    #     print(f"❌ API call failed: {e}")
    #     print("This is expected if you don't have API access configured")
    
    print()
    
    # Example 4: Sequence prediction
    print("=== Sequence Prediction Example ===")
    
    # Create a test sequence
    test_seq = "ATCGATCGATCG" * 170  # ~2KB sequence
    print(f"Test sequence length: {len(test_seq)}bp")
    
    # Note: Actual prediction calls would require valid API access
    # Uncomment the following lines if you have API access:
    
    # try:
    #     print("Running sequence predictions...")
    #     seq_results = core.predict_sequence(
    #         sequence=test_seq,
    #         requested_outputs=["RNA_SEQ", "CAGE"],
    #     )
    #     print(f"✓ Sequence predictions completed:")
    #     print(f"  Output files: {seq_results['output_files'].keys()}")
    #     print(f"  Metadata: {seq_results['metadata_file']}")
    #     
    # except Exception as e:
    #     print(f"❌ API call failed: {e}")
    #     print("This is expected if you don't have API access configured")
    
    print()
    
    # Example 5: Using with different configurations
    print("=== Custom Configuration ===")
    
    custom_config = AlphaGenomeConfig(
        api_key="your_api_key_here",  # Would normally come from environment
        output_dir=Path("custom_output"),
        organism="MUS_MUSCULUS",  # Mouse instead of human
        requested_outputs=["RNA_SEQ", "ATAC", "CHIP_HISTONE"],
        interval_size=50000,  # 50KB intervals for variant analysis
    )
    
    print(f"Custom configuration:")
    print(f"  Organism: {custom_config.organism}")
    print(f"  Output types: {custom_config.requested_outputs}")
    print(f"  Interval size: {custom_config.interval_size}bp")
    print(f"  Output directory: {custom_config.output_dir}")
    
    print("\n=== Example Complete ===")
    print("This demonstrates the core AlphaGenome functionality that can be")
    print("shared between the MCP server and CLI tools.")


if __name__ == "__main__":
    main()