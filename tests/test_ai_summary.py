"""Verify chart grounding, service failures and dashboard summary placement."""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.config import ROOT_DIR
from src.ui.ai_summary import chart_payload, generate_summary
from src.visualization.charts import dashboard_concrete_model_data, dashboard_timber_model_data


@pytest.mark.parametrize("material,loader,names", [
    ("Timber", dashboard_timber_model_data, {"GBM", "MLP"}),
    ("Concrete", dashboard_concrete_model_data, {"RNN", "Ridge"}),
])
def test_payload_matches_chart_data(material, loader, names):
    data = loader()
    payload = json.loads(chart_payload(material, data, "test-pile"))
    assert set(payload["models"]) == names
    assert payload["pile"] == "test-pile"
    for name in names:
        assert len(payload["models"][name]["historical"]) == len(data[name]["historical"])
        assert len(payload["models"][name]["projection"]) == len(data[name]["projection"])


def test_gemini_request_and_response():
    response = Mock()
    response.json.return_value = {"candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"text": "Combined summary"}]},
    }]}
    with patch("src.ui.ai_summary.requests.post", return_value=response) as post:
        assert generate_summary('{"models": {}}', "secret", "gemini-2.5-flash") == (
            "Combined summary"
        )
    assert "secret" not in post.call_args.args[0]
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "secret"


def test_service_error_hides_raw_details():
    response = Mock(status_code=429)
    response.raise_for_status.side_effect = requests.HTTPError("secret", response=response)
    with patch("src.ui.ai_summary.requests.post", return_value=response):
        with pytest.raises(ValueError, match="quota or rate limit") as error:
            generate_summary("{}", "secret", "gemini-2.5-flash")
    assert "secret" not in str(error.value)


def test_truncated_response_is_not_displayed():
    response = Mock()
    response.json.return_value = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
    with patch("src.ui.ai_summary.requests.post", return_value=response):
        with pytest.raises(ValueError, match="complete summary"):
            generate_summary("{}", "secret", "gemini-2.5-flash")


def test_both_pages_have_one_summary_and_cache_by_pile(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch("src.ui.ai_summary.generate_summary", return_value="Combined chart summary") as call:
        app = AppTest.from_file(ROOT_DIR / "app.py").run(timeout=30)
        assert not app.exception
        assert len(app.get("plotly_chart")) == 2
        assert [item.value for item in app.subheader] == ["AI summary"]
        app.run()
        assert call.call_count == 1
        app.sidebar.radio[0].set_value("Concrete").run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 2
        assert [item.value for item in app.subheader] == ["AI summary"]
        assert call.call_count == 2
        app.selectbox[0].select_index(1).run()
        assert not app.exception
        assert call.call_count == 3
