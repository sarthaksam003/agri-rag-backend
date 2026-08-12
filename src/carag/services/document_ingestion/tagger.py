import os
import time
import uuid

import openai
import requests
from chonkie import RecursiveChunker, RecursiveLevel, RecursiveRules  # type: ignore
# from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from carag.core.config import get_settings
from carag.services.document_ingestion.prompt_tagger import (
    HUMAN_TEMPLATE,
    SYSTEMPROMPT_TAGGER,
)

settings = get_settings()
session_id = str(uuid.uuid4())


os.environ["LANGFUSE_SECRET_KEY"] = settings.LF_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LF_PUBLIC_KEY
os.environ["LANGFUSE_BASE_URL"] = settings.LF_HOST


class Questions(BaseModel):
    specific_questions: list[str] = Field(
        description="list of specific questions this chunk addresses"
    )
    general_questions: list[str] = Field(
        description="list of general higher level questions this chunk addresses"
    )


class Tags(BaseModel):
    topic_tag: list[str] = Field(description="list of Topics this chunk covers")
    parent_topic_tag: list[str] = Field(
        description="list of Parent topics this chunk addresses"
    )
    relation_tags: list[str] | None = Field(
        description="How this chunk relates to its surrounding context"
    )
    questions: Questions = Field(description="Questions this chunk can answer")


class LLMSchema(BaseModel):
    tags: list[Tags]


class TagCountMismatchError(ValueError):
    pass


class LLMProcessingError(Exception):
    pass


class AgenticTagger:
    def __init__(
        self,
        llm: BaseChatModel,
        n_chunks_per_batch: int = settings.TAGGER_N_CHUNKS_PER_BATCH,
        tool_choice: str = "LLMSchema",
        template=ChatPromptTemplate(
            [("system", SYSTEMPROMPT_TAGGER), ("human", HUMAN_TEMPLATE)]
        ),
        schema: type[BaseModel] = LLMSchema,
        callbacks: list[BaseCallbackHandler] | None = None,
        max_batch_retries: int = 5,
        batch_retry_wait_seconds: int = 10,
    ):
        if n_chunks_per_batch <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        self.n_chunks_per_batch = n_chunks_per_batch
        # self.runnable = template | create_extractor(
        #     llm, tools=schema, tool_choice=tool_choice
        # )
        self.callbacks = None
        self.runnable = template | llm.with_structured_output(
            schema=schema,
        )
        self.max_batch_retries = max_batch_retries
        self.batch_retry_wait_seconds = batch_retry_wait_seconds
        self.chonkie_recursive_chunker = RecursiveChunker(
            tokenizer_or_token_counter=settings.EMBEDDINGS_API_MODEL,
            chunk_size=1024,
            rules=RecursiveRules(
                [
                    RecursiveLevel(
                        delimiters=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### "],
                        include_delim="next",
                    ),
                    RecursiveLevel(delimiters=["\n\n"]),
                    RecursiveLevel(
                        delimiters=["<table>"],
                        include_delim="next",
                    ),
                ]
            ),
            min_characters_per_chunk=100,
        )

    def _check_vllm_connection(self, vllm_url=settings.OPENAI_API_BASE) -> bool:
        """
        Checks if the vLLM server is reachable.
        You might need to adjust the endpoint based on your vLLM setup.
        """
        try:
            health_url = vllm_url.replace("/v1", "") + "/health"
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM connection check failed: {e}")
            return False

    def _wait_for_vllm(self, delay_seconds=30, tries=10) -> bool:
        num_tries = 0
        while num_tries < tries:
            if self._check_vllm_connection():
                logger.info("vLLM service is available.")
                return True
            num_tries += 1
            logger.info(
                f"""vLLM not ready. Waiting {delay_seconds}s before retry ({num_tries}/{
                    tries
                })..."""
            )
            time.sleep(delay_seconds)
        logger.error(f"vLLM service did not become available after {tries} attempts.")
        return False

    @retry(
        wait=wait_exponential(multiplier=4, min=4, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
        retry=retry_if_exception_type(openai.APIConnectionError),
    )
    def _call_llm(
        self,
        chunk_info: str,
        text_content: str,
        filename: str,
        num_chunks_in_batch: int,
    ) -> LLMSchema:
        try:
            response = self.runnable.with_config(
                {"run_name": "Creating tags", "callbacks": self.callbacks},
            ).invoke(
                {
                    "chunks": chunk_info.strip(),
                    "parent_document": text_content,
                    "num_chunks": num_chunks_in_batch,
                    "filename": filename,
                }
            )
        except Exception as e:
            logger.error(f"LLM call failed with exception: {e}")
            raise LLMProcessingError(f"LLM call failed: {e}") from e

        logger.debug("LLM call successful.")
        return response

    def _flatten_recursive_helper(self, data: dict, results: list[str]) -> None:
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        results.append(f"{key}: {item}")
                    elif isinstance(item, dict):  # Handle nested dicts in lists
                        self._flatten_recursive_helper(item, results)
            elif isinstance(value, dict):
                self._flatten_recursive_helper(value, results)
            elif isinstance(value, str):  # Handle direct string values
                results.append(f"{key}: {value}")

    def _flatten_tags(self, tags_data: dict) -> list[str]:
        flattened_list: list[str] = []
        self._flatten_recursive_helper(tags_data, flattened_list)

        return flattened_list

    @retry(
        wait=wait_exponential(multiplier=4, min=4, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def split_text(self, parent_document: Document) -> list[Document]:
        text_content = parent_document.page_content
        doc_metadata_str = str(parent_document.metadata.get("original_filename"))
        if not text_content or text_content.isspace():
            logger.error(
                f"""Document '{
                    parent_document.metadata
                }' has empty or whitespace-only content."""
            )
            raise ValueError(
                f"""Document '{
                    parent_document.metadata
                }' has empty or whitespace-only content."""
            )

        try:
            initial_chunks: list[str] = [
                chunk.text
                for chunk in self.chonkie_recursive_chunker.chunk(text_content)
            ]
            logger.debug(str(initial_chunks)[:100] + "...")
        except Exception as e:
            logger.error(
                f"""Error during initial chunking of document '{
                    parent_document.metadata
                }': {e}"""
            )
            raise ValueError(
                f"""Error during initial chunking of document '{
                    parent_document.metadata
                }': {e}"""
            ) from e

        batched_chunks = []
        num_chunks = len(initial_chunks)
        for i in range(0, num_chunks, self.n_chunks_per_batch):
            batch = initial_chunks[i : i + self.n_chunks_per_batch]
            batched_chunks.append(batch)

        tagged_documents: list[Document] = []
        for i, batch in enumerate(batched_chunks):
            if not batch:  # Should not happen with correct logic, but safety check
                logger.warning(
                    f"""Encountered an empty batch for document '{
                        parent_document.metadata
                    }' at index {i}. Skipping."""
                )
                continue
            current_batch_size = len(batch)
            chunk_info = ""
            for j, chunk_text in enumerate(batch):
                chunk_info += f"Chunk num: {j}\n"
                chunk_info += f"Chunk_content: {chunk_text}\n"
                chunk_info += "#######\n"

            logger.debug(
                f"""Processing batch {i + 1}/{len(batched_chunks)} for document '{
                    parent_document.metadata
                }'"""
            )

            extracted_schema: LLMSchema | None = None
            batch_success = False
            last_exception = None

            for attempt in range(self.max_batch_retries):
                try:
                    logger.debug(
                        f"""Batch {i + 1}, Attempt {attempt + 1}/{
                            self.max_batch_retries
                        }: Calling LLM..."""
                    )
                    raw_llm_output = self._call_llm(
                        chunk_info,
                        text_content,
                        filename=parent_document.metadata["original_filename"],
                        num_chunks_in_batch=current_batch_size,
                    )

                    extracted_schema = raw_llm_output

                    if len(extracted_schema.tags) != current_batch_size:
                        logger.warning(  # Log as warning during retry attempts
                            f"""Batch {i + 1}, Attempt {attempt + 1}: Mismatch - Got {
                                len(extracted_schema.tags)
                            } tag sets, expected {current_batch_size}."""
                        )
                        raise TagCountMismatchError(
                            f"""Mismatch in batch {i + 1}: Got {
                                len(extracted_schema.tags)
                            } tag sets but expected {current_batch_size}."""
                        )
                    logger.info(
                        f"Batch {i + 1}, Attempt {attempt + 1}: Successfully processed."
                    )
                    batch_success = True
                    break
                except (
                    TagCountMismatchError,
                    LLMProcessingError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                ) as e:
                    logger.warning(
                        f"""Batch {i + 1}, Attempt {attempt + 1} failed: {
                            type(e).__name__
                        } - {e}"""
                    )
                    last_exception = e
                    if attempt < self.max_batch_retries - 1:
                        logger.info(
                            f"""Waiting {
                                self.batch_retry_wait_seconds
                            }s before retrying batch {i + 1}..."""
                        )
                        time.sleep(self.batch_retry_wait_seconds * (2**attempt))
                    else:
                        logger.error(
                            f"""Batch {i + 1} failed permanently after {
                                self.max_batch_retries
                            } attempts for document '{doc_metadata_str}'."""
                        )
                        raise RuntimeError(
                            f"""Failed to process batch {i + 1} for document '{
                                doc_metadata_str
                            }' after {self.max_batch_retries} retries."""
                        ) from last_exception
                except Exception as e:
                    # Catch any other unexpected errors during processing
                    logger.error(
                        f"Batch {i + 1}, Attempt {attempt + 1}: Unexpected error: {e}",
                        exc_info=True,
                    )
                    last_exception = e
                    # Decide if unexpected errors should be retried or fail immediately
                    # Failing immediately is often safer for unknowns.
                    raise RuntimeError(
                        f"""Unexpected error processing batch {i + 1} for document '{
                            doc_metadata_str
                        }'."""
                    ) from e
            if batch_success and extracted_schema and extracted_schema.tags:
                logger.debug(f"Appending results for successful batch {i + 1}")
                for chunk_index_in_batch, (chunk_text, tag_data) in enumerate(
                    zip(batch, extracted_schema.tags)
                ):
                    try:
                        tags_dict = tag_data.model_dump()
                        new_metadata = parent_document.metadata.copy()
                        new_metadata["tags"] = self._flatten_tags(tags_dict)
                        # Add batch and chunk index info for traceability
                        new_metadata["batch_index"] = i
                        new_metadata["chunk_index_in_batch"] = chunk_index_in_batch
                        doc = Document(page_content=chunk_text, metadata=new_metadata)
                        tagged_documents.append(doc)
                    except Exception as e:
                        logger.error(
                            f"""Error processing tags or creating document for chunk {
                                chunk_index_in_batch
                            } in batch {i + 1}: {e}""",
                            exc_info=True,
                        )
                        raise e
                logger.info(
                    f"""Finished processing document '{
                        doc_metadata_str
                    }'. Total tagged documents generated: {len(tagged_documents)}"""
                )
        return tagged_documents

    def split_documents(self, texts: list[Document]) -> list[Document]:
        if not texts:
            raise ValueError("No texts provided, Please Retry")
        if len(texts) == 0:
            raise ValueError("No texts provided, Please Retry")
        final_docs = []

        for doc in texts:
            try:
                chunks = self.split_text(doc)
                if isinstance(chunks, list):
                    final_docs.extend(chunks)
                else:
                    raise ValueError(
                        f"Type chunks mismatch expected list instead got {type(chunks)}"
                    )
            except Exception as e:
                logger.error(f"Failed to process document {doc.metadata}: {e}")
                logger.exception(e)
                continue
        return final_docs


if __name__ == "__main__":
    sample_text = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. There are three main types of machine learning: 
    supervised learning, unsupervised learning, and reinforcement learning.
    
    ## Supervised Learning
    
    Supervised learning uses labeled data to train models. The algorithm learns 
    to map input data to the correct output based on example input-output pairs.
    
    ## Unsupervised Learning
    
    Unsupervised learning finds patterns in unlabeled data. Common techniques 
    include clustering and dimensionality reduction.
    
    <table>
    <tr><th>Algorithm</th><th>Type</th><th>Use Case</th></tr>
    <tr><td>Linear Regression</td><td>Supervised</td><td>Regression</td></tr>
    <tr><td>K-Means</td><td>Unsupervised</td><td>Clustering</td></tr>
    </table>
    
    ### Conclusion
    
    Machine learning has numerous applications across various industries.
    """

    sample_doc = Document(
        page_content=sample_text,
        metadata={"filename": "test_ml_intro.md", "source": "test_source"},
    )

    # Test the tagger
    try:
        # Replace settings with mock for testing
        from carag.core.constants import get_constants

        llm = get_constants().llm

        tagger = AgenticTagger(llm=llm)

        # Test split_text with single document
        print("Testing split_text with sample document...")
        result = tagger.split_text(sample_doc)
        print(f"Generated {len(result)} chunks")

        # Test split_documents with multiple documents
        print("\nTesting split_documents with multiple documents...")
        multiple_docs = [sample_doc, sample_doc]
        results = tagger.split_documents(multiple_docs)
        print(
            f"Generated {len(results)} total chunks from {len(multiple_docs)} documents"
        )

        # Display sample output
        if results:
            print("\nSample chunk metadata:")
            print(f"Content preview: {results[0].page_content[:100]}...")
            print(f"Tags: {results[0].metadata.get('tags', [])[:3]}")  # First 3 tags
            print(f"Batch index: {results[0].metadata.get('batch_index')}")
            print(f"Filename: {results[0].metadata.get('filename')}")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
