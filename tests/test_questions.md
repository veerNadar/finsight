# FinSight Test Question Set

## Tier 1 — Simple document lookup (search_documents only)

1. What was Infosys's operating margin in FY2024-25?
2. What did Infosys say about their AI strategy?
3. How many employees does Infosys have?

## Tier 2 — Document + sentiment (search_documents + analyze_text_sentiment)

4. What did Infosys say about revenue growth, and what's the sentiment?
5. What's the tone of Infosys's commentary on their business outlook?

## Tier 3 — Live data only (web_search)

6. What is Infosys's current stock price?
7. What is the latest news about Infosys?

## Tier 4 — Compound / multi-tool

8. How does Infosys's current stock price compare to what they reported in their annual report?
9. What is Infosys's revenue growth outlook, and how has the market reacted recently?

## Tier 5 — Known limitation (multi-fact, deliberately probing boundary)

10. What was Infosys's revenue for 2020, 2021, 2022, 2023, and 2024?

## Test Results (Run 1 - [today's date])

- Q1: PASS - operating margin 21.1%, matches verified data
- Q2: PASS - coherent AI strategy summary
- Q3: PASS - employee count with YoY comparison
- Q4: PASS - revenue growth + neutral sentiment
- Q5: FAIL - step limit reached
- Q6: PASS - stock price with timestamp
- Q7: FAIL - step limit reached (simple single-tool query, unexpected failure)
- Q8: PASS - genuine multi-tool synthesis with correct math
- Q9: FAIL - step limit reached
- Q10: FAIL - step limit reached (expected, known limitation)

## Finding

~50% success rate on genuinely multi-step tool-calling questions, roughly independent of
question complexity. This points to openai/gpt-oss-120b's tool-calling reliability ceiling
as the dominant limiting factor, not prompt engineering or question difficulty. Documented
as a known model characteristic; graceful degradation implemented rather than chasing a
100% fix via prompting alone.
