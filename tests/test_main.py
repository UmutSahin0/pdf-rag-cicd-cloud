import pytest
from unittest.mock import patch, MagicMock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# PDF yükleme fonksiyonu testi
@patch("app.main.PyPDFLoader")  # <- burada düzeltme yaptık
def test_load_pdf(MockLoader):
    from app.main import load_pdf
    mock_instance = MockLoader.return_value
    mock_instance.load.return_value = [{"page_content": "Mock content"}]

    result = load_pdf("dummy_path.pdf")
    assert isinstance(result, list)
    assert result[0]["page_content"] == "Mock content"


# Split fonksiyonu testi
def test_split_documents():
    from app.main import split_documents

    sample_docs = [Document(page_content="This is a test. Another sentence.")]
    chunks = split_documents(sample_docs, chunk_size=50, chunk_overlap=10)
    
    assert isinstance(chunks, list)
    assert isinstance(chunks[0], Document)
    assert chunks[0].page_content.startswith("This is a test")


# Redis bağlantısı testi
@patch("app.main.RedisClient")  # <- burada düzeltme yaptık
def test_connect_to_redis(MockRedis):
    from app.main import connect_to_redis
    mock_instance = MockRedis.from_url.return_value
    result = connect_to_redis("redis://localhost:6379")
    
    assert result == mock_instance


# Redis index silme testi
@patch("app.main.RedisClient")  # <- burada düzeltme yaptık
def test_clear_index(MockRedis):
    from app.main import clear_index
    mock_instance = MockRedis.from_url.return_value
    mock_ft = mock_instance.ft.return_value
    mock_ft.dropindex.return_value = True

    clear_index(mock_instance, "test_index")
    mock_ft.dropindex.assert_called_with(delete_documents=True)
