import asyncio
import datetime
import json
import math
import sys
import traceback
import uuid
from typing import Literal, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel

from carag.core.config import get_settings
from carag.core.constants import get_constants

settings, constants = get_settings(), get_constants()


class QueryData(TypedDict):
    completed: bool
    learnings: list[str]
    sources: list[dict[str, str]]
    id: str


class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.total_depth: int = depth
        self.total_breadth: int = breadth
        self.current_depth: int = depth
        self.current_breadth: int = 0
        self.queries_by_depth: dict[int, dict[str, QueryData]] = {}
        self.query_order: list[str] = []  # Track order of queries
        self.query_parents: dict[str, str] = {}  # Track parent-child relationships
        self.total_queries: int = 0  # Total number of queries including sub-queries
        self.completed_queries: int = 0
        self.query_ids: dict[str, str] = {}  # Store persistent IDs for queries
        self.root_query: str | None = None  # Store the root query

    async def start_query(
        self, query: str, depth: int, parent_query: str | None = None
    ):
        """Record the start of a new query"""
        # Generate a unique ID for this query
        query_id = str(uuid.uuid4())
        self.query_ids[query] = query_id

        # If this is the first query, set it as the root
        if self.root_query is None:
            self.root_query = query

        # Initialize the depth level if it doesn't exist
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        # Add the query to the appropriate depth level if it's not already there
        if query not in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "sources": [],  # Add sources list to store source information
                "id": self.query_ids[query],  # Use persistent ID
            }
            self.query_order.append(query)
            if parent_query:
                self.query_parents[query] = parent_query
            self.total_queries += 1

        self.current_depth = depth
        self.current_breadth = len(self.queries_by_depth[depth])
        await self._report_progress("query_started")

    async def add_learning(self, query: str, depth: int, learning: str):
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query]["learnings"].append(learning)
            await self._report_progress("learning_added")

    async def complete_query(self, query: str, depth: int):
        """Mark a query as completed"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                await self._report_progress(f"Completed query: {query}")

                # Check if parent query exists and update its status if all children are complete
                parent_query = self.query_parents.get(query)
                if parent_query:
                    await self._update_parent_status(parent_query)

    async def add_sources(self, query: str, depth: int, sources: list[dict[str, str]]):
        # TODO: Modify for working with chunkID rather than sources.
        """Record sources for a specific query"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            # Add new sources that aren't already in the list
            current_sources = self.queries_by_depth[depth][query]["sources"]
            current_urls = {source["url"] for source in current_sources}

            for source in sources:
                if source["url"] not in current_urls:
                    current_sources.append(source)
                    current_urls.add(source["url"])

            await self._report_progress(f"Added sources for query: {query}")

    async def _update_parent_status(self, parent_query: str):
        """Update parent query status based on children completion"""
        # Find all children of this parent
        children = [q for q, p in self.query_parents.items() if p == parent_query]

        # Check if all children are complete
        parent_depth = next(
            (
                d
                for d, queries in self.queries_by_depth.items()
                if parent_query in queries
            ),
            None,
        )

        if parent_depth is not None:
            all_children_complete = all(
                self.queries_by_depth[d][q]["completed"]
                for q in children
                for d in self.queries_by_depth
                if q in self.queries_by_depth[d]
            )

            if all_children_complete:
                # Complete the parent query
                await self.complete_query(parent_query, parent_depth)

    async def _report_progress(self, action: str):
        """Report current progress and stream to client if callback provided"""
        # Build event data for streaming
        progress_data = {
            "type": "research_progress",
            "action": action,
            "timestamp": datetime.datetime.now().isoformat(),
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
            "progress_percentage": int(
                (self.completed_queries / max(1, self.total_queries)) * 100
            ),
        }

        # Add tree structure if root query exists
        if self.root_query:
            progress_data["tree"] = self._build_research_tree()

        # logger.info progress to console
        logger.info(
            f"[Progress] {action}: {progress_data['progress_percentage']}% complete"
        )

    def _build_research_tree(self):
        """Build a tree structure of the research queries"""

        def build_node(query):
            """Recursively build the tree node"""
            # Find the depth for this query
            depth = next(
                (d for d, queries in self.queries_by_depth.items() if query in queries),
                0,
            )

            data = self.queries_by_depth[depth][query]

            # Find all children of this query
            children = [q for q, p in self.query_parents.items() if p == query]

            return {
                "query": query,
                "id": self.query_ids[query],
                "status": "completed" if data["completed"] else "in_progress",
                "depth": depth,
                "learnings": data["learnings"],
                "sources": data["sources"],  # Include sources in the tree
                "sub_queries": [build_node(child) for child in children],
                "parent_query": self.query_parents.get(query),
            }

        # Start building from the root query
        if self.root_query:
            return build_node(self.root_query)
        return {}

    def get_learnings_by_query(self):
        """Get all learnings organized by query"""
        learnings = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                if data["learnings"]:
                    learnings[query] = data["learnings"]
        return learnings


class DeepSearch:
    def __init__(
        self,
        api_key: str,
        mode: Literal["fast", "balanced", "comprehensive"] = "balanced",
    ):
        """
        Initialize DeepSearch with a mode parameter:
        - "fast": Prioritizes speed (reduced breadth/depth, highest concurrency)
        - "balanced": Default balance of speed and comprehensiveness
        - "comprehensive": Maximum detail and coverage
        """
        self.llm: BaseChatModel = constants.llm
        self.query_history: set[str] = set()
        self.mode: Literal["fast", "balanced", "comprehensive"] = mode

    def determine_research_breadth_and_depth(self, query: str):
        """Determine the appropriate research breadth and depth based on the query complexity"""

        class ResearchParameters(BaseModel):
            """
            - "breadth": integer between 1-10 (number of parallel search queries)
            - "depth": integer between 1-5 (levels of follow-up questions)
            - "explanation": brief explanation of your reasoning

            """

            breadth: int
            depth: int
            explanation: str

        user_prompt = """
        Analyze this research query and determine the appropriate breadth (number of parallel search queries) 
        and depth (levels of follow-up questions) needed for thorough research:

        Query: {query}

        Consider:
        1. Complexity of the topic
        2. Breadth of knowledge required
        3. Depth of expertise needed
        4. Potential for follow-up exploration
        """
        template = ChatPromptTemplate([("user", user_prompt)])

        try:
            chain = template | self.llm.with_structured_output(ResearchParameters)
            response: ResearchParameters = cast(
                ResearchParameters, chain.invoke({"query": query})
            )

            return {
                "breadth": response.breadth,
                "depth": response.depth,
                "explanation": response.explanation,
            }

        except Exception as e:
            logger.error(f"Error determining research parameters: {str(e)}")

            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)

            defaults = {
                "fast": {"breadth": 3, "depth": 1},
                "balanced": {"breadth": 5, "depth": 2},
                "comprehensive": {"breadth": 7, "depth": 3},
            }
            return defaults.get(
                self.mode,
                {"breadth": 5, "depth": 2, "explanation": "Using default values."},
            )

    def generate_follow_up_questions(
        self,
        query: str,
        max_questions: int = 5,
    ):
        """Generate follow-up questions based on the initial query"""

        class FollowUpQuestions(BaseModel):
            "Follow up questions to clarify user intent"

            follow_up_queries: list[str]

        user_prompt = """
        Based on the following user query, generate upto {max_questions} follow-up questions that would help clarify what the user wants to know.
		These questions should:
		1. Seek to understand the user's specific information needs
		2. Clarify ambiguous terms or concepts in the original query
		3. Determine the scope or boundaries of what the user is looking for
		4. Identify the user's level of familiarity with the topic
		5. Uncover the user's purpose or goal for seeking this information

		User Query: {query}
        """
        template = ChatPromptTemplate([{"system": user_prompt}])

        try:
            chain = template | self.llm.with_structured_output(FollowUpQuestions)

            response = chain.invoke({"query": query, "max_questions": max_questions})

            try:
                return cast(FollowUpQuestions, response).follow_up_queries

            except Exception as e:
                logger.error(f"Error parsing follow-up questions: {str(e)}")

                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                tb_text = "".join(tb_lines)
                logger.error(tb_text)
                raise e

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            return [f"What are the key aspects of {query}?"]

    async def generate_queries(
        self,
        query: str,
        num_queries: int = 3,
        learnings: list[str] = [],
        previous_queries: set[str] | None = None,  # Add previous_queries parameter
    ):
        """Generate search queries based on the initial query and learnings"""
        if previous_queries is None:
            previous_queries = set()

        # Adjust the prompt based on the mode
        prompt_by_mode = {
            "fast": "Generate concise, focused search queries",
            "balanced": "Generate balanced search queries that explore different aspects",
            "comprehensive": "Generate comprehensive search queries that deeply explore the topic",
        }

        mode_prompt = prompt_by_mode.get(self.mode, prompt_by_mode["balanced"])

        # Format learnings for the prompt
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])
        learnings_section = (
            f"\nBased on what we've learned so far:\n{learnings_text}"
            if learnings
            else ""
        )

        # Format previous queries for the prompt
        previous_queries_text = "\n".join([f"- {q}" for q in previous_queries])
        previous_queries_section = (
            f"\nPrevious search queries (avoid repeating these):\n{previous_queries_text}"
            if previous_queries
            else ""
        )

        user_prompt = """
        You are a research assistant helping to explore the topic: "{query}"
        
        {mode_prompt}.
        {learnings_section}
        {previous_queries_section}
        
        Generate {num_queries} specific search queries that would help gather comprehensive information about this topic.
        Each query should focus on a different aspect or subtopic.
        Make the queries specific and well-formed for a search engine.
        
        Format your response as a JSON object with a "queries" field containing an array of query strings.
        """
        template = ChatPromptTemplate([("system", user_prompt)])

        class QueryResponse(BaseModel):
            """Queries"""

            queries: list[str]

        try:
            chain = template | self.llm.with_structured_output(QueryResponse)

            response = chain.invoke(
                {
                    "query": query,
                    "mode_prompt": mode_prompt,
                    "learnings_section": learnings_section,
                    "previous_queries_section": previous_queries_section,
                    "num_queries": num_queries,
                }
            )
            try:
                parsed_response = cast(QueryResponse, response)
                queries = set(parsed_response.queries)

                # Filter out any queries that are too similar to previous ones
                unique_queries = set()
                for q in queries:
                    is_similar = False
                    for prev_q in previous_queries:
                        if await self._are_queries_similar(q, prev_q):
                            is_similar = True
                            break

                    if not is_similar:
                        unique_queries.add(q)

                return unique_queries
            except Exception as e:
                logger.error(f"Error parsing query response: {str(e)}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                tb_text = "".join(tb_lines)
                logger.error(tb_text)
                raise e

        except Exception as e:
            logger.info(f"Error generating queries: {str(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            # Fallback to basic queries if generation fails
            return {f"{query} - aspect {i + 1}" for i in range(num_queries)}

    def format_text_with_sources(self, response_dict: dict, answer: str):
        """
        Format text with sources from Gemini response, adding citations at specified positions.
        Returns tuple of (formatted_text, sources_dict).
        """
        if not response_dict or not response_dict.get("candidates"):
            return answer, {}

        # Get grounding metadata from the response
        grounding_metadata = response_dict["candidates"][0].get("grounding_metadata")
        if not grounding_metadata:
            return answer, {}

        # Get grounding chunks and supports
        grounding_chunks = grounding_metadata.get("grounding_chunks", [])
        grounding_supports = grounding_metadata.get("grounding_supports", [])

        if not grounding_chunks or not grounding_supports:
            return answer, {}

        try:
            # Create mapping of URLs
            sources = {
                i: {
                    "link": chunk.get("web", {}).get("uri", ""),
                    "title": chunk.get("web", {}).get("title", ""),
                }
                for i, chunk in enumerate(grounding_chunks)
                if chunk.get("web")
            }

            # Create a list of (position, citation) tuples
            citations = []
            for support in grounding_supports:
                segment = support.get("segment", {})
                indices = support.get("grounding_chunk_indices", [])

                if indices and segment and segment.get("end_index") is not None:
                    end_index = segment["end_index"]
                    source_idx = indices[0]
                    if source_idx in sources:
                        citation = (
                            f"[[{source_idx + 1}]]({sources[source_idx]['link']})"
                        )
                        citations.append((end_index, citation))

            # Sort citations by position (end_index)
            citations.sort(key=lambda x: x[0])

            # Insert citations into the text
            result = ""
            last_pos = 0
            for pos, citation in citations:
                result += answer[last_pos:pos]
                result += citation
                last_pos = pos

            # Add any remaining text
            result += answer[last_pos:]

            return result, sources

        except Exception as e:
            logger.info(f"Error processing grounding metadata: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            return answer, {}

    async def search(self, query: str):
        # TODO: Replace with vectorDB
        model_id = "gemini-2.0-flash"

        google_search_tool = types.Tool(google_search=types.GoogleSearch())

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
            "response_modalities": ["TEXT"],
            "tools": [google_search_tool],
        }

        response = await self.client.aio.models.generate_content(
            model=model_id, contents=query, config=generation_config
        )

        response_dict = response.model_dump()

        formatted_text, sources = self.format_text_with_sources(
            response_dict, response.text
        )

        return formatted_text, sources

    async def process_result(
        self,
        query: str,
        result: str,
        num_learnings: int = 3,
        num_follow_up_questions: int = 3,
    ):
        """Process search results to extract learnings and generate follow-up questions"""

        class ProcessedResult(BaseModel):
            """
            "learnings": array of learning strings
            "follow_up_questions": array of question strings
            """

            learnings: list[str]
            follow_up_questions: list[str]

        user_prompt = """
        Analyze the following search results for the query: "{query}"
        
        Search Results:
        {result}
        
        Please extract:
        1. The {num_learnings} most important learnings or insights from these results
        2. {num_follow_up_questions} follow-up questions that would help explore this topic further
        """

        template = ChatPromptTemplate([("system", user_prompt)])

        try:
            chain = template | self.llm.with_structured_output(ProcessedResult)
            response = chain.invoke(
                {
                    "query": query,
                    "result": result,
                    "num_learnings": num_learnings,
                    "num_follow_up_questions": num_follow_up_questions,
                }
            )

            try:
                # Get the parsed response using the Pydantic model
                parsed_response = cast(ProcessedResult, response)
                return {
                    "learnings": parsed_response.learnings,
                    "follow_up_questions": parsed_response.follow_up_questions,
                }
            except Exception as e:
                logger.info(f"Error parsing process_result: {str(e)}")
                # Fallback to generating follow-up questions separately
                follow_up_questions = self.generate_follow_up_questions(
                    query, num_follow_up_questions
                )
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                tb_text = "".join(tb_lines)
                logger.error(tb_text)

                # Extract some basic learnings from the result
                learnings = [
                    f"Information about {query}",
                    f"Details related to {query}",
                ]

                return {
                    "learnings": learnings,
                    "follow_up_questions": follow_up_questions,
                }

        except Exception as e:
            logger.info(f"Error processing result: {str(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            return {
                "learnings": [f"Information about {query}"],
                "follow_up_questions": [f"What are the key aspects of {query}?"],
            }

    async def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are semantically similar"""
        # Simple string comparison for exact matches
        if query1.lower() == query2.lower():
            return True

        # For very short queries, use substring check
        if len(query1) < 10 or len(query2) < 10:
            return query1.lower() in query2.lower() or query2.lower() in query1.lower()

        # For more complex queries, use Gemini to check similarity
        class SimilarityResult(BaseModel):
            """Class for indicating similarity"""

            are_similar: bool

        user_prompt = """
        Compare these two search queries and determine if they are semantically similar 
        (would likely return similar search results):
        
        Query 1: {query1}
        Query 2: {query2}
        
        Return a JSON object with a single boolean field "are_similar" indicating if the queries are similar.
        """
        template = ChatPromptTemplate([("system", user_prompt)])

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "response_schema": SimilarityResult,
        }

        chain = template | self.llm.with_structured_output(SimilarityResult)

        response = chain.invoke(
            {
                "query1": query1,
                "query2": query2,
            }
        )
        # Get the parsed response using the Pydantic model
        parsed_response = cast(SimilarityResult, response)
        return parsed_response.are_similar

    async def deep_research(
        self,
        query: str,
        breadth: int,
        depth: int,
        learnings: list[str] = [],
        visited_urls: dict[int, dict] = {},
        parent_query: str = None,
    ):
        progress = ResearchProgress(depth, breadth)

        # Start the root query
        await progress.start_query(query, depth, parent_query)

        # Adjust number of queries based on mode
        max_queries = {
            "fast": 5,
            "balanced": 10,
            "comprehensive": 7,  # Changed from 15 to 7
        }[self.mode]

        queries = await self.generate_queries(
            query,
            min(breadth, max_queries),
            learnings,
            previous_queries=self.query_history,
        )

        self.query_history.update(queries)
        unique_queries = list(queries)[:breadth]

        async def process_query(query_str: str, current_depth: int, parent: str = None):
            try:
                # Start this query as a sub-query of the parent
                await progress.start_query(query_str, current_depth, parent)

                result = await self.search(query_str)
                # The search method returns a tuple (formatted_text, sources)
                formatted_text, new_urls = result

                # Add sources to the progress tracker for this query
                if new_urls:
                    sources_list = [
                        {"url": url_data["link"], "title": url_data["title"]}
                        for url_data in new_urls.values()
                        if "link" in url_data and "title" in url_data
                    ]
                    await progress.add_sources(query_str, current_depth, sources_list)

                processed_result = await self.process_result(
                    query=query_str,
                    result=formatted_text,
                    num_learnings=min(5, math.ceil(breadth / 1.5)),
                    num_follow_up_questions=min(5, math.ceil(breadth / 1.5)),
                )

                # Record learnings
                for learning in processed_result["learnings"]:
                    await progress.add_learning(query_str, current_depth, learning)

                new_urls = result[1]
                max_idx = max(visited_urls.keys()) if visited_urls else -1
                all_urls = {
                    **visited_urls,
                    **{(i + max_idx + 1): url_data for i, url_data in new_urls.items()},
                }

                # Only go deeper if in comprehensive mode and depth > 1
                if self.mode == "comprehensive" and current_depth > 1:
                    # Reduced breadth for deeper levels, but increased from previous implementation
                    # Less aggressive reduction
                    new_breadth = min(5, math.ceil(breadth / 1.5))
                    new_depth = current_depth - 1

                    # Select most important follow-up questions instead of just one
                    if processed_result["follow_up_questions"]:
                        # Take up to 3 most relevant questions instead of just 1
                        follow_up_questions = processed_result["follow_up_questions"][
                            :3
                        ]

                        # Process each sub-query
                        for next_query in follow_up_questions:
                            sub_results = await process_query(
                                next_query,
                                new_depth,
                                query_str,  # Pass current query as parent
                            )

                            # Merge the sub-results with the current results
                            if sub_results:
                                # Add sub-query learnings to all_urls
                                if "visited_urls" in sub_results:
                                    for url_key, url_data in sub_results[
                                        "visited_urls"
                                    ].items():
                                        if url_data["link"] not in [
                                            u["link"] for u in all_urls.values()
                                        ]:
                                            max_idx = (
                                                max(all_urls.keys()) if all_urls else -1
                                            )
                                            all_urls[max_idx + 1] = url_data

                await progress.complete_query(query_str, current_depth)
                return {
                    "learnings": processed_result["learnings"],
                    "visited_urls": all_urls,
                }

            except Exception as e:
                logger.info(f"Error processing query {query_str}: {str(e)}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                tb_text = "".join(tb_lines)
                logger.error(tb_text)
                await progress.complete_query(query_str, current_depth)
                return {"learnings": [], "visited_urls": {}}

        # Process queries concurrently
        tasks = [process_query(q, depth, query) for q in unique_queries]
        results = await asyncio.gather(*tasks)

        # Combine results
        all_learnings = list(
            set(learning for result in results for learning in result["learnings"])
        )

        all_urls = {}
        current_idx = 0
        seen_urls = set()
        for result in results:
            for url_data in result["visited_urls"].values():
                if url_data["link"] not in seen_urls:
                    all_urls[current_idx] = url_data
                    seen_urls.add(url_data["link"])
                    current_idx += 1

        # Complete the root query after all sub-queries are done
        await progress.complete_query(query, depth)

        # Build the research tree
        research_tree = progress._build_research_tree()

        logger.info(f"Research tree built with {len(all_learnings)} learnings")
        # save the research tree to a file
        with open("research_tree.json", "w") as f:
            json.dump(research_tree, f)

        return {
            "learnings": all_learnings,
            "visited_urls": all_urls,
            "tree": research_tree,  # Return the tree structure
        }

    async def generate_final_report(
        self, query: str, learnings: list[str], visited_urls: dict[int, dict]
    ) -> str:
        # Format learnings for the prompt
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])

        user_prompt = """
		You are a creative storyteller tasked with transforming research into an engaging and distinctive report.

        Research Query: {query}

        Key Discoveries:
        {learnings_text}

        Craft a captivating report that:

        # CREATIVE APPROACH
        1. Opens with an imaginative introduction that draws readers into the topic
        2. Transforms key discoveries into a compelling narrative with your unique voice
        3. Adds fresh perspectives and unexpected connections between ideas
        4. Experiments freely with tone, style, and expression
        5. Concludes with thought-provoking reflections that linger with the reader

        # FORMATTING TOOLS
        - Create evocative, imaginative headings
        - Use markdown formatting (##, ###, **bold**, *italics*) for visual interest
        - Incorporate blockquotes for emphasis or contrast
        - Deploy bullet points or numbered lists where they enhance clarity
        - Insert tables to organize information in visually appealing ways
        - Use horizontal rules (---) to create dramatic pauses or section breaks

        Feel free to be bold, experimental, and expressive while maintaining clarity and coherence. There are no academic conventions to follow - let your creativity flow!
        """
        template = ChatPromptTemplate([("system", user_prompt)])

        try:
            chain = template | self.llm
            response = chain.invoke(
                {
                    "query": query,
                    "learnings_text": learnings_text,
                }
            )

            return response.content
        except Exception as e:
            logger.info(f"Error generating final report: {str(e)}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            return f"Error generating report: {str(e)}"
