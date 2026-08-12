from datetime import datetime
from uuid import uuid4

from carag.api.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    DocumentIngestionResponse,
)


class TestChatSchemas:
    def test_chat_message_request_valid(self):
        request = ChatMessageRequest(message="Hello, world!", session_id="test-session")
        assert request.message == "Hello, world!"
        assert request.session_id == "test-session"

    def test_chat_message_response_valid(self):
        response = ChatMessageResponse(
            id=str(uuid4()),
            session_id="test-session",
            message="Hello",
            response="Hi there!",
            timestamp=datetime.now(),
            hallucination_check={"score": 0.9},
            retrieved_documents_count=3,
        )
        assert response.message == "Hello"
        assert response.response == "Hi there!"
        assert response.retrieved_documents_count == 3


class TestDocumentSchemas:
    def test_document_ingestion_response_valid(self):
        response = DocumentIngestionResponse(
            document_id=str(uuid4()),
            filename="test.pdf",
            status="completed",
            chunks_processed=10,
            total_chunks=10,
            processing_time=2.5,
        )
        assert response.filename == "test.pdf"
        assert response.status == "completed"
        assert response.chunks_processed == 10
