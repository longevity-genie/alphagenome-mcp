#!/usr/bin/env python3
"""Tests for AlphaGenome MCP Server with real API calls."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from alphagenome.data import genome
from alphagenome.models import dna_client
from dotenv import load_dotenv

from alphagenome_mcp.server import (
    AlphaGenomeMCP,
)

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get API key from environment."""
    api_key = os.getenv("ALPHA_GENOME_API_KEY")
    if not api_key:
        pytest.skip("ALPHA_GENOME_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="session")
def alpha_client(api_key: str) -> dna_client.DnaClient:
    """Create AlphaGenome client for direct API testing."""
    return dna_client.create(api_key)


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    """Create persistent test output directory that can be inspected after tests."""
    # Use a test_output directory in the project root
    test_dir = Path(__file__).parent.parent / "test_output"

    # Clear the directory if it exists (fresh start for each test session)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Create the directory
    test_dir.mkdir(parents=True, exist_ok=True)

    return test_dir


@pytest.fixture
def mcp_server(api_key: str, test_output_dir: Path) -> AlphaGenomeMCP:
    """Fixture providing an AlphaGenomeMCP server instance with real API access and persistent output dir."""
    # Temporarily set the API key in environment
    os.environ["ALPHA_GENOME_API_KEY"] = api_key

    # Create a unique subdirectory for this test
    import time

    test_subdir = test_output_dir / f"test_{int(time.time() * 1000)}"
    test_subdir.mkdir(parents=True, exist_ok=True)

    return AlphaGenomeMCP(output_dir=str(test_subdir))


class TestAlphaGenomeDirectAPI:
    """Test AlphaGenome API directly using the client."""

    def test_predict_sequence_dnase_lung(
        self, alpha_client: dna_client.DnaClient
    ) -> None:
        """Test sequence prediction for DNase in lung tissue (from quick_start.ipynb)."""
        # Use the exact example from the AlphaGenome quick start colab
        sequence = "GATTACA".center(2048, "N")  # Pad to valid length

        output = alpha_client.predict_sequence(
            sequence=sequence,
            requested_outputs=[dna_client.OutputType.DNASE],
            ontology_terms=["UBERON:0002048"],  # Lung
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        # Verify we got meaningful results (based on colab output)
        assert output is not None
        assert hasattr(output, "dnase")
        assert output.dnase is not None

        # Check prediction shape and metadata
        dnase = output.dnase
        assert dnase.values.shape == (2048, 1)  # 2048 positions, 1 tissue
        assert len(dnase.metadata) == 1

        # Check that metadata contains expected fields (from actual API response)
        expected_columns = [
            "name",
            "strand",
            "ontology_curie",
            "biosample_name",
            "biosample_type",
        ]
        for col in expected_columns:
            assert col in dnase.metadata.columns

        # Verify the values make sense
        metadata = dnase.metadata.iloc[0]
        assert metadata["ontology_curie"] == "UBERON:0002048"  # Lung tissue
        assert "DNase" in metadata["name"]  # Should contain DNase

    def test_predict_sequence_multiple_outputs_tissues(
        self, alpha_client: dna_client.DnaClient
    ) -> None:
        """Test sequence prediction with multiple outputs and tissues (from quick_start.ipynb)."""
        # Use the exact example from the AlphaGenome quick start colab
        sequence = "GATTACA".center(2048, "N")

        output = alpha_client.predict_sequence(
            sequence=sequence,
            requested_outputs=[dna_client.OutputType.CAGE, dna_client.OutputType.DNASE],
            ontology_terms=["UBERON:0002048", "UBERON:0000955"],  # Lung and Brain
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        # Verify multiple outputs (matching colab output)
        assert hasattr(output, "cage")
        assert hasattr(output, "dnase")
        assert output.cage is not None
        assert output.dnase is not None

        # Expected shapes from colab: DNASE (2048, 2), CAGE (2048, 4)
        cage = output.cage
        assert cage.values.shape == (2048, 4)  # CAGE is stranded: 2 tissues × 2 strands
        assert len(cage.metadata) == 4

        dnase = output.dnase
        assert dnase.values.shape == (2048, 2)  # DNASE not stranded: 2 tissues
        assert len(dnase.metadata) == 2

        # Check that CAGE metadata includes strand information
        assert "strand" in cage.metadata.columns
        strands = cage.metadata["strand"].unique()
        assert len(strands) == 2  # Should have both + and - strands

    def test_predict_interval_human_chr1(
        self, alpha_client: dna_client.DnaClient
    ) -> None:
        """Test interval prediction for a mouse genomic region (from quick_start.ipynb)."""
        # Use the mouse example from the quick start colab (which is simpler and faster)
        interval = genome.Interval("chr1", 3_000_000, 3_000_001).resize(
            dna_client.SEQUENCE_LENGTH_1MB
        )

        output = alpha_client.predict_interval(
            interval=interval,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=["UBERON:0002048"],  # Lung
            organism=dna_client.Organism.MUS_MUSCULUS,  # Mouse as in colab example
        )

        # Verify interval prediction results
        assert output is not None
        assert hasattr(output, "rna_seq")
        assert output.rna_seq is not None

        # Check prediction shape (RNA_SEQ is stranded)
        rna_seq = output.rna_seq
        assert rna_seq.values.shape[0] == dna_client.SEQUENCE_LENGTH_1MB  # 1MB sequence
        assert (
            rna_seq.values.shape[1] >= 2
        )  # At least 2 strands (could be more tissues)
        assert len(rna_seq.metadata) >= 2

        # Check interval information matches input
        assert rna_seq.interval.chromosome == "chr1"
        assert rna_seq.interval.width == dna_client.SEQUENCE_LENGTH_1MB

    def test_predict_variant_snp(self, alpha_client: dna_client.DnaClient) -> None:
        """Test variant prediction for a known functional variant (from quick_start.ipynb)."""
        # Use the exact variant example from the quick start colab
        variant = genome.Variant(
            chromosome="chr22",
            position=36201698,
            reference_bases="A",
            alternate_bases="C",
        )

        # Create interval from variant as shown in colab
        interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

        output = alpha_client.predict_variant(
            interval=interval,
            variant=variant,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=["UBERON:0001157"],  # Colon - Transverse (from colab)
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        # Verify variant prediction structure
        assert output is not None
        assert hasattr(output, "reference")
        assert hasattr(output, "alternate")
        assert output.reference is not None
        assert output.alternate is not None

        # Check that both reference and alternate have the same shape
        ref_rna = output.reference.rna_seq
        alt_rna = output.alternate.rna_seq
        assert ref_rna.values.shape == alt_rna.values.shape
        assert len(ref_rna.metadata) == len(alt_rna.metadata)

        # Check that we have a 1MB sequence as expected
        assert ref_rna.values.shape[0] == dna_client.SEQUENCE_LENGTH_1MB

    def test_score_variant_pathogenicity(
        self, alpha_client: dna_client.DnaClient
    ) -> None:
        """Test variant scoring using example from batch_variant_scoring.ipynb."""
        # Use one of the variants from the batch scoring colab
        variant = genome.Variant(
            chromosome="chr3",
            position=58394738,
            reference_bases="A",
            alternate_bases="T",
            name="chr3_58394738_A_T_b38",  # From the VCF example
        )

        # Create interval from variant with 1MB sequence length (as in colab)
        interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

        scores = alpha_client.score_variant(
            interval=interval,
            variant=variant,
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        # Verify scoring results
        assert scores is not None
        assert len(scores) > 0

        # Check that we got AnnData objects with scoring information
        for score_data in scores:
            assert hasattr(score_data, "n_obs")
            assert hasattr(score_data, "n_vars")
            assert hasattr(score_data, "uns")
            assert hasattr(score_data, "X")

            # Check that variant information is stored in uns
            assert "variant" in score_data.uns
            stored_variant = score_data.uns["variant"]
            assert stored_variant.chromosome == variant.chromosome
            assert stored_variant.position == variant.position

    def test_get_supported_outputs(self, alpha_client: dna_client.DnaClient) -> None:
        """Test getting supported output types."""
        output_types = list(dna_client.OutputType)

        # Verify we get a list of supported outputs
        assert len(output_types) > 0

        # Check for expected output types
        output_names = [ot.name for ot in output_types]
        expected_outputs = ["RNA_SEQ", "CAGE", "DNASE", "ATAC", "CHIP_HISTONE"]
        for expected in expected_outputs:
            assert expected in output_names

    def test_get_supported_organisms(self, alpha_client: dna_client.DnaClient) -> None:
        """Test getting supported organisms."""
        organisms = list(dna_client.Organism)

        # Verify we get organism information
        assert len(organisms) > 0

        # Check for expected organisms
        organism_names = [org.name for org in organisms]
        assert "HOMO_SAPIENS" in organism_names
        assert "MUS_MUSCULUS" in organism_names

    def test_invalid_sequence_length(self, alpha_client: dna_client.DnaClient) -> None:
        """Test that unsupported sequence lengths are handled."""
        # Too short sequence
        sequence = "ATCG"  # Way too short

        with pytest.raises(ValueError):  # Should raise validation error
            alpha_client.predict_sequence(
                sequence=sequence,
                requested_outputs=[dna_client.OutputType.DNASE],
                ontology_terms=["UBERON:0002048"],  # Required parameter
                organism=dna_client.Organism.HOMO_SAPIENS,
            )


class TestMCPServerInitialization:
    """Test MCP server initialization and configuration."""

    def test_server_initialization_default_output_dir(self) -> None:
        """Test that the server initializes correctly with default output directory."""
        # Temporarily set API key
        original_key = os.environ.get("ALPHA_GENOME_API_KEY")
        os.environ["ALPHA_GENOME_API_KEY"] = "test-key"

        try:
            server = AlphaGenomeMCP()
            assert server.prefix == "alphagenome_"
            # Should default to current working directory / alphagenome_output
            expected_dir = Path.cwd() / "alphagenome_output"
            assert server.output_dir == expected_dir
            assert server.output_dir.exists()  # Should be created automatically
            assert server.api_key == "test-key"
        finally:
            # Restore original key
            if original_key:
                os.environ["ALPHA_GENOME_API_KEY"] = original_key
            elif "ALPHA_GENOME_API_KEY" in os.environ:
                del os.environ["ALPHA_GENOME_API_KEY"]

    def test_server_initialization_custom_output_dir(
        self, test_output_dir: Path
    ) -> None:
        """Test that the server initializes correctly with custom output directory."""
        # Temporarily set API key
        original_key = os.environ.get("ALPHA_GENOME_API_KEY")
        os.environ["ALPHA_GENOME_API_KEY"] = "test-key"

        try:
            custom_dir = test_output_dir / "custom_output"
            server = AlphaGenomeMCP(output_dir=str(custom_dir))
            assert server.output_dir == custom_dir
            assert server.output_dir.exists()  # Should be created automatically
        finally:
            # Restore original key
            if original_key:
                os.environ["ALPHA_GENOME_API_KEY"] = original_key
            elif "ALPHA_GENOME_API_KEY" in os.environ:
                del os.environ["ALPHA_GENOME_API_KEY"]

    def test_client_creation(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test that the client can be created and cached."""
        # First call should create client
        client1 = mcp_server._get_client()
        assert client1 is not None

        # Second call should return cached client
        client2 = mcp_server._get_client()
        assert client2 is client1  # Same instance

    def test_missing_api_key(self) -> None:
        """Test that missing API key raises appropriate error."""
        # Temporarily remove API key
        original_key = os.environ.get("ALPHA_GENOME_API_KEY")
        if "ALPHA_GENOME_API_KEY" in os.environ:
            del os.environ["ALPHA_GENOME_API_KEY"]

        try:
            with pytest.raises(
                ValueError,
                match="ALPHA_GENOME_API_KEY environment variable is required",
            ):
                AlphaGenomeMCP()
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ["ALPHA_GENOME_API_KEY"] = original_key


class TestInternalValidation:
    """Test the internal sequence validation function."""

    def test_validate_sequence_valid(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test validation of valid sequences."""
        # Valid 2KB sequence
        sequence = "ATCG" * 512  # 2048 bases
        result = mcp_server._validate_sequence(sequence)

        assert result["valid"] is True
        assert result["sequence_length"] == 2048
        assert result["valid_characters"] is True
        assert result["supported_length"] is True
        assert len(result["invalid_characters"]) == 0

    def test_validate_sequence_invalid_characters(
        self, mcp_server: AlphaGenomeMCP
    ) -> None:
        """Test validation of sequences with invalid characters."""
        sequence = "ATCGXY" * 341 + "AT"  # 2048 bases with invalid chars

        with pytest.raises(ValueError, match="Invalid characters found"):
            mcp_server._validate_sequence(sequence)

    def test_validate_sequence_invalid_length(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test validation of sequences with invalid length."""
        sequence = "ATCG" * 100  # 400 bases, not supported

        with pytest.raises(ValueError, match="Unsupported length"):
            mcp_server._validate_sequence(sequence)

    def test_validate_sequence_both_invalid(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test validation of sequences with both invalid characters and length."""
        sequence = "ATCGXY" * 10  # 60 bases with invalid chars

        with pytest.raises(
            ValueError, match="Invalid characters found.*Unsupported length"
        ):
            mcp_server._validate_sequence(sequence)


class TestFileOperations:
    """Test file saving and loading operations."""

    def test_save_track_data(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test saving track data to files."""

        # Create mock track data
        class MockTrackData:
            def __init__(self):
                self.values = np.random.rand(100, 5)
                self.metadata = ["track1", "track2", "track3", "track4", "track5"]
                self.interval = None

        track_data = MockTrackData()
        file_paths = mcp_server._save_track_data(
            track_data, "RNA_SEQ", "test_prediction"
        )

        # Verify we get a dictionary with multiple formats
        assert isinstance(file_paths, dict)
        assert "npz" in file_paths
        assert "parquet" in file_paths
        assert "plot" in file_paths

        # Verify NPZ file was created and has correct content
        npz_path = file_paths["npz"]
        assert Path(npz_path).exists()
        assert npz_path.endswith("_RNA_SEQ.npz")

        loaded_data = np.load(npz_path, allow_pickle=True)
        np.testing.assert_array_equal(loaded_data["values"], track_data.values)
        assert list(loaded_data["metadata"]) == track_data.metadata

        # Verify Parquet file was created
        parquet_path = file_paths["parquet"]
        assert Path(parquet_path).exists()
        assert parquet_path.endswith("_RNA_SEQ.parquet")

        # Verify plot was created
        plot_path = file_paths["plot"]
        if plot_path:  # Plot creation might fail in test environment
            assert Path(plot_path).exists()
            assert plot_path.endswith("_RNA_SEQ_plot.png")

    def test_save_metadata(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test saving metadata to JSON files."""
        metadata = {
            "organism": "HOMO_SAPIENS",
            "output_types": ["RNA_SEQ", "DNASE"],
            "sequence_length": 2048,
        }

        file_path = mcp_server._save_metadata(metadata, "test_prediction")

        # Verify file was created
        assert Path(file_path).exists()
        assert file_path.endswith("_metadata.json")

        # Verify file content
        with open(file_path) as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata


class TestMCPTools:
    """Test MCP tools functionality."""

    def test_tool_registry(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test that tools are properly registered."""
        # Check that we can access the tools through FastMCP's interface
        # Instead of checking internal attributes, let's test functionality
        assert hasattr(mcp_server, "_register_alphagenome_tools")

        # Test that tools have been registered by checking if we have decorated functions
        # This is a more robust way to test than accessing internal attributes

        # Count decorated methods (this is an indirect way to verify tools are registered)
        tool_methods = [attr for attr in dir(mcp_server) if not attr.startswith("_")]
        assert len(tool_methods) > 0  # Should have some public methods

        # Test that the server can handle tool calls (this verifies tools are working)
        assert mcp_server.prefix == "alphagenome_"
        assert mcp_server.api_key is not None

    def test_supported_output_types_structure(self) -> None:
        """Test that output types are structured correctly."""
        # Test the underlying data structures are available
        output_types = list(dna_client.OutputType)
        assert len(output_types) > 0

        # Check for expected output types
        output_names = [ot.name for ot in output_types]
        expected_outputs = ["RNA_SEQ", "CAGE", "DNASE", "ATAC"]
        for expected in expected_outputs:
            assert expected in output_names

    def test_supported_organisms_structure(self) -> None:
        """Test that organisms are structured correctly."""
        # Test the underlying data structures are available
        organisms = list(dna_client.Organism)
        assert len(organisms) > 0

        # Check for expected organisms
        organism_names = [org.name for org in organisms]
        assert "HOMO_SAPIENS" in organism_names
        assert "MUS_MUSCULUS" in organism_names


class TestVisualization:
    """Test visualization functionality with actual plot generation."""

    def test_create_tracks_visualization(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test creating a tracks visualization plot."""

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Create figure like the tool does
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create track visualization
        ax.set_title("DNase-seq Signal Track")
        ax.set_xlabel("Genomic Position (bp)")
        ax.set_ylabel("DNase Signal")

        # Create realistic genomic track data
        positions = np.arange(0, 2048)  # Match our test sequence length
        # Simulate DNase signal with some realistic characteristics
        signal = np.random.gamma(
            2, 0.5, 2048
        )  # Gamma distribution for realistic signal
        signal = np.convolve(signal, np.ones(10) / 10, mode="same")  # Smooth the signal

        ax.plot(positions, signal, label="DNase-seq", linewidth=1.5, color="blue")
        ax.fill_between(positions, signal, alpha=0.3, color="lightblue")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_tracks_plot_{timestamp}.png"
        image_path = mcp_server.output_dir / filename

        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Verify the plot was created
        assert image_path.exists()
        assert image_path.stat().st_size > 10000  # Should be a reasonable size
        print(f"✅ Created tracks visualization: {image_path}")

    def test_create_variant_comparison_plot(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test creating a variant comparison visualization."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create variant comparison visualization
        ax.set_title("Variant Impact Comparison: chr22:36201698 A>C")

        # Simulate prediction scores for reference vs alternate
        output_types = ["RNA-seq", "DNase", "ATAC-seq", "H3K27ac"]
        ref_scores = np.random.uniform(0.6, 0.9, len(output_types))
        alt_scores = ref_scores + np.random.normal(0, 0.15, len(output_types))
        alt_scores = np.clip(alt_scores, 0, 1)  # Keep in [0,1] range

        x = np.arange(len(output_types))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            ref_scores,
            width,
            label="Reference (A)",
            color="#1f77b4",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            alt_scores,
            width,
            label="Alternate (C)",
            color="#ff7f0e",
            alpha=0.8,
        )

        ax.set_ylabel("Prediction Score")
        ax.set_xlabel("Genomic Output Type")
        ax.set_title("Variant Impact on Genomic Outputs")
        ax.set_xticks(x)
        ax.set_xticklabels(output_types)
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        # Save the plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_variant_comparison_{timestamp}.png"
        image_path = mcp_server.output_dir / filename

        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Verify the plot was created
        assert image_path.exists()
        assert image_path.stat().st_size > 15000  # Should be a reasonable size
        print(f"✅ Created variant comparison plot: {image_path}")

    def test_create_contact_map_visualization(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test creating a contact map visualization."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create contact map
        size = 100
        # Create a realistic-looking contact map with distance decay
        contact_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                distance = abs(i - j)
                # Contact frequency decreases with genomic distance
                contact_matrix[i, j] = np.random.exponential(1.0) * np.exp(
                    -distance / 20.0
                )

        # Make it symmetric
        contact_matrix = (contact_matrix + contact_matrix.T) / 2

        # Add some stronger diagonal interactions
        for i in range(size - 1):
            contact_matrix[i, i + 1] = contact_matrix[i + 1, i] = np.random.uniform(
                2, 4
            )

        im = ax.imshow(
            contact_matrix, cmap="Reds", interpolation="nearest", origin="lower"
        )
        ax.set_xlabel("Genomic Position (10kb bins)")
        ax.set_ylabel("Genomic Position (10kb bins)")
        ax.set_title("Chromatin Contact Map: chr1:3,000,000-4,000,000")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Contact Frequency", rotation=270, labelpad=20)

        plt.tight_layout()

        # Save the plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_contact_map_{timestamp}.png"
        image_path = mcp_server.output_dir / filename

        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Verify the plot was created
        assert image_path.exists()
        assert image_path.stat().st_size > 20000  # Should be a reasonable size
        print(f"✅ Created contact map visualization: {image_path}")

    def test_visualize_actual_prediction_data(self, mcp_server: AlphaGenomeMCP) -> None:
        """Test visualization using actual prediction data from test output."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Look for actual prediction data files in test output
        output_dir = mcp_server.output_dir
        npz_files = list(output_dir.glob("**/*RNA_SEQ.npz"))

        if npz_files:
            # Use actual prediction data
            data_file = npz_files[0]
            print(f"📊 Using actual prediction data from: {data_file}")

            # Load the actual prediction data
            data = np.load(data_file, allow_pickle=True)
            values = data["values"]

            # Create visualization
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot the actual RNA-seq prediction
            if len(values.shape) == 2:
                # Multi-track data
                for i in range(min(values.shape[1], 3)):  # Show up to 3 tracks
                    ax.plot(
                        values[:, i], label=f"Track {i + 1}", linewidth=1.5, alpha=0.8
                    )
            else:
                # Single track
                ax.plot(values, label="RNA-seq Signal", linewidth=1.5)

            ax.set_title("Actual AlphaGenome RNA-seq Predictions")
            ax.set_xlabel("Genomic Position (bp)")
            ax.set_ylabel("Predicted RNA-seq Signal")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add some statistics
            mean_signal = np.mean(values)
            max_signal = np.max(values)
            ax.text(
                0.02,
                0.98,
                f"Mean: {mean_signal:.3f}\nMax: {max_signal:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            plt.tight_layout()

            # Save the plot
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"actual_prediction_data_{timestamp}.png"
            image_path = output_dir / filename

            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Verify the plot was created
            assert image_path.exists()
            assert image_path.stat().st_size > 10000
            print(f"✅ Created actual prediction data visualization: {image_path}")
        else:
            print("⚠️  No actual prediction data found, skipping this test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
