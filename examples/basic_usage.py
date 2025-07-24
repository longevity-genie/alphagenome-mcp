#!/usr/bin/env python3
"""
Basic usage example for AlphaGenome MCP Server

This example demonstrates how to use the AlphaGenome MCP server
to make predictions and analyze genomic sequences.
"""

import asyncio
import json

# This would typically be done via MCP client, but shown here for demonstration
from alphagenome_mcp.server import AlphaGenomeMCP, SequencePredictionRequest


async def basic_prediction_example():
    """Example showing basic sequence prediction."""

    # Initialize the MCP server (in practice, this would be done by MCP client)
    mcp = AlphaGenomeMCP(output_dir="./examples_output")

    # Example DNA sequence (short example)
    sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    print(f"Making prediction for sequence: {sequence}")

    # Create prediction request
    request = SequencePredictionRequest(
        sequence=sequence,
        output_types=["gene_expression", "chromatin_features"]
    )

    try:
        # This simulates what the MCP client would do
        result = await mcp.predict_sequence(request.sequence, request.output_types)

        print("Prediction completed successfully!")
        print(f"Result summary: {json.dumps(result, indent=2)}")

    except Exception as e:
        print(f"Prediction failed: {e}")


def interval_prediction_example():
    """Example showing interval-based prediction."""

    print("\n" + "="*50)
    print("Interval Prediction Example")
    print("="*50)

    # Example genomic interval
    interval_info = {
        "chromosome": "chr1",
        "start": 1000000,
        "end": 1001000,
        "genome_build": "hg38"
    }

    print(f"Predicting for interval: {interval_info}")
    print("This would use the predict_interval MCP tool in practice.")


def variant_analysis_example():
    """Example showing variant effect prediction."""

    print("\n" + "="*50)
    print("Variant Analysis Example")
    print("="*50)

    # Example variant
    variant_info = {
        "chromosome": "chr7",
        "position": 117199644,
        "reference": "C",
        "alternate": "T",
        "context_length": 1000
    }

    print(f"Analyzing variant: {variant_info}")
    print("This would use the predict_variant and score_variant MCP tools.")


if __name__ == "__main__":
    print("AlphaGenome MCP Server - Basic Usage Examples")
    print("=" * 60)

    # Run the async example
    asyncio.run(basic_prediction_example())

    # Show other examples (these would need actual MCP client integration)
    interval_prediction_example()
    variant_analysis_example()

    print("\n" + "="*60)
    print("Examples completed!")
    print("\nTo use these features with an MCP client:")
    print("1. Set up your ALPHAGENOME_API_KEY environment variable")
    print("2. Start the MCP server: uvx alphagenome-mcp-stdio")
    print("3. Connect your MCP client using the provided configuration files")
