"""AlphaGenome MCP Server - Interface for Google DeepMind's AlphaGenome genomics predictions."""

from alphagenome_mcp.server import AlphaGenomeMCP, app

__version__ = "0.1.0"
__all__ = ["app", "AlphaGenomeMCP", "main"]

def main() -> None:
    """Main entry point for the CLI."""
    app()
