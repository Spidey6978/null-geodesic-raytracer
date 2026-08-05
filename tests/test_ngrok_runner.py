"""
Module: tests.test_ngrok_runner
Automated unit tests for public server runner and ngrok configuration.
"""

from unittest.mock import patch, MagicMock
from scripts.run_public_server import start_public_server


def test_start_public_server_mocked():
    mock_ngrok = MagicMock()
    mock_ngrok.connect.return_value.public_url = "https://mock-blackhole.ngrok-free.app"

    with patch("scripts.run_public_server.ngrok", mock_ngrok), \
         patch("uvicorn.run") as mock_uvicorn:

        start_public_server(port=8000, auth_token="test_token")

        mock_ngrok.set_auth_token.assert_called_with("test_token")
        mock_ngrok.connect.assert_called_with(8000, "http")
        mock_uvicorn.assert_called_once()
