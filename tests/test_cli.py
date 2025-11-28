import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from grimoire.cli import app
from pathlib import Path
import json

runner = CliRunner()

def test_process_file_command_success(mock_config, mock_key_manager, mock_genai_client):
    # Mock core.process_single_file to avoid actual processing logic in CLI test
    with patch("grimoire.core.process_single_file") as mock_process:
        mock_process.return_value = {"status": "success", "file": "test.pdf", "message": "Done"}
        
        # Create dummy file
        with runner.isolated_filesystem():
            Path("test.pdf").touch()
            result = runner.invoke(app, ["process-file", "test.pdf", "--json"])
            
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["status"] == "success"

def test_process_file_not_found():
    result = runner.invoke(app, ["process-file", "nonexistent.pdf", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["status"] == "error"
    assert "File not found" in data["message"]

def test_get_summary_command(mock_config):
    with patch("grimoire.core.get_summary_json") as mock_get:
        mock_get.return_value = {"header": {"title": "Test Book"}}
        
        result = runner.invoke(app, ["get-summary", "test_book", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["header"]["title"] == "Test Book"

def test_ask_command(mock_config):
    with patch("grimoire.core.ask_question") as mock_ask:
        mock_ask.return_value = "This is the answer."
        
        result = runner.invoke(app, ["ask", "What is magic?", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["answer"] == "This is the answer."
