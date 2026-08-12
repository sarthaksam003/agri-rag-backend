from fastapi import status


class TestDocumentEndpoints:
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"

    def test_list_documents_empty(self, client, tenant_headers):
        """Test listing documents when none exist"""
        response = client.get("/api/v1/documents/", headers=tenant_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Documents retrieved successfully"
        assert "data" in data

    def test_ingest_single_pdf(self, client, tenant_headers, sample_pdf_1):
        """Test ingesting a single PDF"""
        with open(sample_pdf_1, "rb") as f:
            files = [("files", (sample_pdf_1.name, f, "application/pdf"))]
            response = client.post(
                "/api/v1/documents/ingest", headers=tenant_headers, files=files
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_processed"] >= 1
        assert data["total_failed"] == 0

        # Check individual file result
        file_result = data["processed_files"][0]
        assert file_result["filename"] == sample_pdf_1.name
        assert file_result["status"] == "completed"
        assert file_result["chunks_processed"] > 0

    def test_ingest_multiple_pdfs(
        self, client, tenant_headers, sample_pdf_1, sample_pdf_2
    ):
        """Test ingesting multiple PDFs in one request"""
        files = []
        for pdf_path in [sample_pdf_1, sample_pdf_2]:
            files.append(
                ("files", (pdf_path.name, open(pdf_path, "rb"), "application/pdf"))
            )

        try:
            response = client.post(
                "/api/v1/documents/ingest", headers=tenant_headers, files=files
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_processed"] == 2
            assert data["total_failed"] == 0

            # Verify both files were processed
            filenames = [f["filename"] for f in data["processed_files"]]
            assert sample_pdf_1.name in filenames
            assert sample_pdf_2.name in filenames

        finally:
            # Close file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()

    def test_ingest_invalid_file_type(self, client, tenant_headers):
        """Test uploading non-PDF files"""
        # Create a temporary text file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as f:
            f.write("This is a text file, not PDF")
            f.flush()

            with open(f.name, "rb") as file_handle:
                files = [("files", ("test.txt", file_handle, "text/plain"))]
                response = client.post(
                    "/api/v1/documents/ingest", headers=tenant_headers, files=files
                )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_delete_all_tenant_documents(self, client, tenant_headers):
        """Test deleting all documents for a tenant"""
        response = client.delete("/api/v1/documents/tenant/all", headers=tenant_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "deleted_chunks" in data
        assert data["tenant_id"] == "demo-tenant"
        assert "successfully" in data["message"].lower()

    def test_missing_tenant_header(self, client):
        """Test requests without tenant header"""
        response = client.get("/api/v1/documents/")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
