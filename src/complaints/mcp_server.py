"""Door for AI agents: the predictor exposed as MCP tools.

Run with: uv run python -m complaints.mcp_server        (stdio transport)
Needs the `mcp` dependency group: uv sync --group mcp

Tools:
    classify_complaint(text)  ->  product, confidence, needs_review, probabilities
    list_products()           ->  the classes the model can return, and the threshold
"""

from __future__ import annotations

from complaints.predictor import get_predictor


def classify_complaint(text: str) -> dict:
    """Route a banking complaint to a product team from its text.

    Returns the predicted product, the model's confidence, and needs_review,
    which is true when the confidence is below the serving threshold and a
    person should decide instead.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    return get_predictor().predict(text).to_dict()


def list_products() -> dict:
    """The product classes the classifier can return, and the review threshold in use."""
    predictor = get_predictor()
    return {
        "model": predictor.model_name,
        "products": predictor.classes,
        "review_threshold": predictor.review_threshold,
    }


def build_server():
    """Register the two tools on a FastMCP server. Imported lazily so the package
    works without the optional `mcp` dependency."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("banking-complaints")
    server.tool()(classify_complaint)
    server.tool()(list_products)
    return server


if __name__ == "__main__":
    build_server().run()
