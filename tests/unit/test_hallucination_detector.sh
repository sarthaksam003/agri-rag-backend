#!/bin/bash

# RAG Hallucination Detection Test Suite
# Run these tests one by one and analyze the responses

echo "================================"
echo "Test 1: Perfect Match (No Hallucinations)"
echo "Expected: High consistency, 0% hallucination"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Python is a high-level programming language. It was created by Guido van Rossum in 1991.",
    "chunks": [
      "Python is a high-level, interpreted programming language known for its simplicity.",
      "Python was created by Guido van Rossum and first released in 1991."
    ],
    "chunk_topics": ["python", "programming"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 2: Complete Hallucination"
echo "Expected: Low consistency, 100% hallucination"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "The Moon is made of cheese. Elephants can fly. Water boils at 50 degrees Celsius.",
    "chunks": [
      "The Moon is Earth natural satellite composed primarily of rock and metal.",
      "Elephants are large terrestrial mammals that cannot fly.",
      "Water boils at 100 degrees Celsius at sea level atmospheric pressure."
    ],
    "chunk_topics": ["moon", "elephants", "water"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 3: Partial Hallucination - Wrong Numbers"
echo "Expected: Medium consistency, some hallucination on numeric facts"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "The human body has 500 bones. The heart pumps blood throughout the body. Adults need about 15 hours of sleep per night.",
    "chunks": [
      "The adult human body has 206 bones.",
      "The heart is a muscular organ that pumps blood through the circulatory system.",
      "Most adults need 7-9 hours of sleep per night for optimal health."
    ],
    "chunk_topics": ["human anatomy", "heart", "sleep"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 4: Unsupported Claims (No Chunks)"
echo "Expected: Should detect fabricated information"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Machine learning is a subset of AI. Deep learning uses neural networks. Transformers were invented in 2017. GPT-4 has 100 trillion parameters.",
    "chunks": [
      "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
      "Deep learning is a type of machine learning that uses artificial neural networks with multiple layers."
    ],
    "chunk_topics": ["machine learning", "deep learning"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 5: Contradictory Information"
echo "Expected: High hallucination on contradicted facts"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Albert Einstein was born in 1850. He developed the theory of relativity. He won the Nobel Prize in Chemistry.",
    "chunks": [
      "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
      "Einstein is best known for developing the theory of relativity.",
      "Einstein was awarded the Nobel Prize in Physics in 1921."
    ],
    "chunk_topics": ["einstein", "relativity", "nobel prize"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 6: Exaggeration/Embellishment"
echo "Expected: Moderate hallucination on exaggerated claims"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "The Amazon rainforest is the largest rainforest on Earth, covering the entire South American continent. It produces all of the worlds oxygen and contains every species of plant known to science.",
    "chunks": [
      "The Amazon rainforest is the largest tropical rainforest in the world, covering approximately 5.5 million square kilometers.",
      "The Amazon produces about 20% of the worlds oxygen.",
      "The Amazon contains approximately 10% of all species on Earth."
    ],
    "chunk_topics": ["amazon rainforest", "biodiversity"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 7: Mixed Content (Some chunks unused)"
echo "Expected: Should detect underutilized chunks"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Tokyo is the capital of Japan. It has a population of over 13 million people.",
    "chunks": [
      "Tokyo is the capital and largest city of Japan.",
      "Tokyo has a population of approximately 14 million in the city proper.",
      "Mount Fuji is located about 100 kilometers southwest of Tokyo.",
      "Tokyo hosted the Summer Olympics in 1964 and 2020.",
      "The Tokyo Metro is one of the busiest subway systems in the world."
    ],
    "chunk_topics": ["tokyo", "population", "mount fuji", "olympics", "metro"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 8: Contextual Hallucination"
echo "Expected: Should detect incorrect context/attribution"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Shakespeare wrote Romeo and Juliet in 1623. It was his first play and was performed at the Globe Theatre in London.",
    "chunks": [
      "Romeo and Juliet was written by William Shakespeare around 1594-1596.",
      "The Globe Theatre was built in 1599 and was the primary venue for Shakespeares plays.",
      "Romeo and Juliet is one of Shakespeares most famous tragedies, but not his first work."
    ],
    "chunk_topics": ["shakespeare", "romeo and juliet", "globe theatre"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 9: Edge Case - Empty Response"
echo "Expected: Should handle gracefully"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "",
    "chunks": [
      "This is a chunk of information.",
      "This is another chunk."
    ],
    "chunk_topics": ["topic1", "topic2"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 10: Edge Case - Single Word Claims"
echo "Expected: Should handle very short responses"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Yes. Correct. True.",
    "chunks": [
      "The answer to the question is affirmative.",
      "This statement is correct and verified."
    ],
    "chunk_topics": ["confirmation"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "Test 11: Require All Topics = True"
echo "Expected: Should enforce all topics are covered"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "Python is a programming language. It is easy to learn.",
    "chunks": [
      "Python is a high-level programming language.",
      "Python is known for its simple syntax.",
      "JavaScript is another popular programming language.",
      "Java is used for enterprise applications."
    ],
    "chunk_topics": ["python", "javascript", "java"],
    "require_all_topics": true
  }'
echo -e "\n\n"

echo "================================"
echo "Test 12: Technical Jargon Accuracy"
echo "Expected: Should detect incorrect technical terms"
echo "================================"
curl -X POST "http://localhost:14787/detect-rag-hallucinations" \
    -H "Content-Type: application/json" \
    -d '{
    "llm_response": "HTTP uses port 443 for secure connections. TCP is a connectionless protocol. DNS resolves domain names to MAC addresses.",
    "chunks": [
      "HTTPS uses port 443 for secure connections, while HTTP uses port 80.",
      "TCP (Transmission Control Protocol) is a connection-oriented protocol.",
      "DNS (Domain Name System) resolves domain names to IP addresses."
    ],
    "chunk_topics": ["networking", "protocols", "dns"],
    "require_all_topics": false
  }'
echo -e "\n\n"

echo "================================"
echo "TESTS COMPLETE"
echo "================================"
echo ""
echo "CONFIGURATION RECOMMENDATIONS:"
echo ""
echo "Based on test results, consider these thresholds:"
echo "1. overall_consistency_score < 0.5 → High risk, reject response"
echo "2. hallucination_ratio > 0.3 → Moderate risk, flag for review"
echo "3. hallucination_ratio > 0.5 → High risk, reject response"
echo "4. Check underutilized_chunks → If >50% unused, consider retrieval quality"
echo "5. For critical applications, use stricter thresholds (0.7+ consistency)"
echo ""
echo "Suggested Actions:"
echo "- consistency < 0.3: 'I don't have reliable information about that.'"
echo "- consistency 0.3-0.5: 'I found some information, but I'm not fully confident.'"
echo "- consistency 0.5-0.7: Provide answer with mild disclaimer"
echo "- consistency > 0.7: Provide answer confidently"
echo "- hallucination_ratio > 0.5: Regenerate or refuse"
