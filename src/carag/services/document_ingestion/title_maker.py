import os

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from carag.core.config import get_settings

settings = get_settings()
os.environ["LANGFUSE_SECRET_KEY"] = settings.LF_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LF_PUBLIC_KEY
os.environ["LANGFUSE_BASE_URL"] = settings.LF_HOST

langfuse_handler = CallbackHandler()


class LLMOutput(BaseModel):
    """A dictionary mapping title IDs to their hierarchical levels."""

    title_mapping: dict[int, int] = Field(
        ...,
        description="Dictionary mapping title indices to their hierarchical levels (1-4)",
    )


class Titler:
    def __init__(self):
        SYSTEMPROMPT = """Reasoning:Low
The input is a dictionary composed of all titles from a document. Please optimize the titles according to the following guidelines to make the results conform to the normal document hierarchy:
1. Each value in the dictionary is a list containing the following elements:
    - Title text
    - Text line height is the average line height of the block where the title is located
    - Page number where the title is located

2. Preserve original content:
    - All elements in the input dictionary are valid and cannot be deleted
    - Please ensure that the number of elements in the output dictionary matches the input

3. Keep the key-value correspondence in the dictionary unchanged

4. Optimize hierarchical structure:
    - Add appropriate hierarchical structure to each title element
    - Titles with larger line heights are generally higher-level titles
    - Title levels from front to back must be continuous, levels cannot be skipped
    - Title levels should be at most 4 levels, do not add too many levels
    - The optimized title should only retain the integer representing the level of the title, do not retain other information
    
5. Handle problematic titles:
    - If a title appears to be repeated (e.g., "MODULE - 2" appearing multiple times), mark only the first occurrence as its appropriate level (1-3) and mark subsequent identical or near-identical occurrences as level 4
    - If a title breaks the document flow (e.g., "Do you know", "ACTIVITY 12.1"), mark it as level 4
    - For titles that are instructional or activity-based rather than content-structural, mark them as level 4
    - If a title is just a decorative element or doesn't contribute to the document structure, mark it as level 4

6. Maintain logical document structure:
    - Ensure that each level 1 heading represents a major section of the document
    - Level 2 headings should be meaningful subsections of their parent level 1 heading
    - Level 3 headings should be meaningful subsections of their parent level 2 heading
    - Use level 4 for minor headings, activities, examples, or other non-structural elements
    - Avoid creating "orphaned" headings (e.g., a single level 2 heading without siblings)
    
7. Reasonableness check and fine-tuning:
    - After completing the initial classification, carefully check the reasonableness of the classification results
    - Based on context relationships and logical order, adjust unreasonable classifications
    - Ensure that the final classification results conform to the actual structure and logic of the document
    - The dictionary may contain body text that has been mistaken for titles, you can exclude them by marking their level as 0

IMPORTANT: 
Please return ONLY a valid JSON object with a single key "title_mapping" that contains a dictionary mapping title indices to their levels.
    """  # noqa: E501

        HUMANPROMPT = """Input Title list:
{input_dict}

Return the optimized title mapping as JSON:
"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_API_MODEL,
            base_url=settings.OPENAI_API_BASE,
            temperature=0.1,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", SYSTEMPROMPT), ("human", HUMANPROMPT)]
        )
        self.__initialize_bounded_llm()

    def __initialize_bounded_llm(self):
        structured_llm = self.llm.with_structured_output(LLMOutput)
        self.bounded_llm = self.prompt_template | structured_llm

    def merge_para_with_text(self, block: dict) -> str:
        """Extracts and concatenates all text content from a block's spans."""
        text_parts = []
        for line in block["lines"]:
            for span in line["spans"]:
                if "content" in span:
                    text_parts.append(span["content"])
        return "".join(text_parts)

    @retry(stop=stop_after_attempt(5), retry=retry_if_exception_type(ValidationError))
    def invoke_titler(self, title_dict: dict["str", "str"]):
        completion = self.bounded_llm.invoke(
            {"input_dict": title_dict},
            config={"callbacks": [langfuse_handler]},
        )
        return completion

    def llm_aided_title(self, pdf_info_list):
        title_dict = {}
        origin_title_list = []
        i = 0
        for page in pdf_info_list:
            blocks = page["para_blocks"]
            page_idx = page["page_idx"]
            for block in blocks:
                if block["type"] == "title":
                    origin_title_list.append(block)
                    title_text = self.merge_para_with_text(block)
                    page_line_height_list = []
                    for line in block["lines"]:
                        bbox = line["bbox"]
                        page_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(page_line_height_list) > 0:
                        line_avg_height = sum(page_line_height_list) / len(
                            page_line_height_list
                        )
                    else:
                        line_avg_height = int(block["bbox"][3] - block["bbox"][1])
                    title_dict[f"{i}"] = [
                        title_text,
                        line_avg_height,
                        page_idx + 1,
                    ]
                    i += 1
        if not title_dict:
            logger.warning("No titles found to process.")
            return pdf_info_list

        retry_count = 0
        max_retries = 3
        dict_completion = None

        while retry_count < max_retries:
            try:
                completion = self.invoke_titler(title_dict)
                dict_completion = completion.title_mapping

                # Process successful completion
                processed_indices = set()
                for i, origin_title_block in enumerate(origin_title_list):
                    if i in dict_completion:
                        origin_title_block["level"] = int(dict_completion[i])
                        processed_indices.add(i)

                # Check if any titles were missed
                if len(processed_indices) == len(origin_title_list):
                    logger.info("Successfully processed all titles")
                    break
                else:
                    missing_count = len(origin_title_list) - len(processed_indices)
                    logger.warning(
                        f"LLM skipped {missing_count} header levels. "
                        f"Processed {len(processed_indices)} out of {len(origin_title_list)} titles. Retrying..."
                    )
                    retry_count += 1

            except Exception as e:
                logger.exception(f"Error during title processing: {e}")
                retry_count += 1

        # After retries: only assign level 3 to missing ones
        if dict_completion is not None:
            for i, origin_title_block in enumerate(origin_title_list):
                if "level" not in origin_title_block or i not in dict_completion:
                    origin_title_block["level"] = 3
                    logger.debug(f"Assigned level 3 to missing title index {i}")
        else:
            # Complete failure case: assign level 3 to all
            logger.error(
                "Failed to process titles after maximum retries, assigning level 3 to all"
            )
            for origin_title_block in origin_title_list:
                origin_title_block["level"] = 3

        return pdf_info_list
