"""
Automated tests for FastAPI endpoints and response consistency.
Verifies health, image routes, segmentation overlays, and ViT attention endpoints.
"""
import pytest
from fastapi import HTTPException
from unittest.mock import patch

from app.api.routes import (
    health_check,
    get_attention_image,
    get_segmentation_image,
    get_analysis_view,
    router,
)


def test_api_health_check():
    """Verify health check endpoint returns status ok."""
    data = health_check()
    assert data["status"] == "ok"
    assert "CEFM Backend" in data["message"]


def test_routes_registered_in_router():
    """Verify essential endpoints are properly registered in FastAPI router."""
    routes = {r.path for r in router.routes}
    assert "/health" in routes
    assert "/images/{analysis_id}/attention" in routes
    assert "/images/{analysis_id}/segmentation" in routes
    assert "/analyses/{analysis_id}/view/{view_name}" in routes


def test_attention_endpoint_not_found_on_fake_id():
    """Verify attention endpoint raises 404 on nonexistent analysis ID."""
    with patch("app.api.routes.analyses_collection.find_one", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            get_attention_image("nonexistent_fake_analysis_id")
        assert exc_info.value.status_code == 404


def test_segmentation_endpoint_not_found_on_fake_id():
    """Verify segmentation endpoint raises 404 on nonexistent analysis ID."""
    with patch("app.api.routes.analyses_collection.find_one", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            get_segmentation_image("nonexistent_fake_analysis_id")
        assert exc_info.value.status_code == 404


def test_view_attention_endpoint_not_found_on_fake_id():
    """Verify XAI view endpoint raises 404 on nonexistent analysis ID."""
    with patch("app.api.routes.analyses_collection.find_one", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            get_analysis_view("nonexistent_fake_analysis_id", "attention")
        assert exc_info.value.status_code == 404
