from fastapi import status


class TestCompleteWorkflow:
    def test_complete_rag_workflow(
        self, client, tenant_headers, sample_pdf_1, sample_pdf_2
    ):
        """Test complete workflow: ingest → chat → query documents"""

        # Step 1: Ingest documents
        files = []
        for pdf_path in [sample_pdf_1, sample_pdf_2]:
            files.append(
                ("files", (pdf_path.name, open(pdf_path, "rb"), "application/pdf"))
            )

        try:
            # Ingest both PDFs
            ingest_response = client.post(
                "/api/v1/documents/ingest", headers=tenant_headers, files=files
            )
            assert ingest_response.status_code == status.HTTP_200_OK
            ingest_data = ingest_response.json()
            assert ingest_data["total_processed"] == 2

            # Get document IDs from ingestion response
            document_ids = [
                doc["document_id"] for doc in ingest_data["processed_files"]
            ]

            # Step 2: List documents to verify ingestion
            list_response = client.get("/api/v1/documents/", headers=tenant_headers)
            assert list_response.status_code == status.HTTP_200_OK
            list_data = list_response.json()
            assert list_data["data"]["total_documents"] > 0

            # Step 3: Send chat message about the documents
            chat_data = {
                "message": "Can you summarize what these documents are about?",
                "session_id": "e2e-test-session",
            }

            chat_response = client.post(
                "/api/v1/chat/message", headers=tenant_headers, json=chat_data
            )
            assert chat_response.status_code == status.HTTP_200_OK
            chat_result = chat_response.json()

            # Verify chat response contains meaningful content
            assert len(chat_result["response"]) > 10
            assert chat_result["retrieved_documents_count"] > 0

            # Step 4: Send follow-up question in same session
            followup_data = {
                "message": "Tell me more about the main topics",
                "session_id": "e2e-test-session",
            }

            followup_response = client.post(
                "/api/v1/chat/message", headers=tenant_headers, json=followup_data
            )
            assert followup_response.status_code == status.HTTP_200_OK

            # Step 5: Delete one specific document
            if document_ids:
                delete_response = client.delete(
                    f"/api/v1/documents/{document_ids[0]}", headers=tenant_headers
                )
                assert delete_response.status_code == status.HTTP_200_OK

            # Step 6: Clean up - delete all tenant data
            cleanup_response = client.delete(
                "/api/v1/documents/tenant/all", headers=tenant_headers
            )
            assert cleanup_response.status_code == status.HTTP_200_OK

        finally:
            # Close file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()

    def test_chat_without_documents(self, client, tenant_headers):
        """Test chat functionality when no documents are ingested"""
        chat_data = {
            "message": "What can you tell me about the documents?",
            "session_id": "no-docs-session",
        }

        response = client.post(
            "/api/v1/chat/message", headers=tenant_headers, json=chat_data
        )

        # Should still work, but might say no information available
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["response"]) > 0
