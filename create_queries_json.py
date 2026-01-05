"""
Create Queries JSON for KL Divergence Computation
=================================================
This script helps you create a queries.json file that can be used with 
compute_kl_divergence.py

Usage:
    python create_queries_json.py --output queries.json
"""

import json
import os


def create_queries_json(output_path: str, questions: list, document_paths: list = None):
    """
    Create a queries JSON file.
    
    Args:
        output_path: Where to save the JSON file
        questions: List of question dictionaries
        document_paths: Optional list of document paths to attach to all questions
    """
    queries = []
    
    for q in questions:
        query = {
            "question": q["question"],
        }
        
        if "expected_answer" in q:
            query["expected_answer"] = q["expected_answer"]
        
        # Add documents if specified
        if "documents" in q:
            query["documents"] = q["documents"]
        elif document_paths:
            query["documents"] = []
            for doc_path in document_paths:
                if doc_path.endswith('.txt'):
                    query["documents"].append({"type": "txt", "path": doc_path})
        
        queries.append(query)
    
    output_data = {"queries": queries}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created queries file: {output_path}")
    print(f"  - {len(queries)} questions")
    return output_path


# Questions WITHOUT document context (model uses its own knowledge)
QUESTIONS_NO_CONTEXT = [
    {
        "question": "What is machine learning?",
    },
    {
        "question": "Explain neural networks.",
    },
]

# Questions WITH document context (from extracted PDF text files)
QUESTIONS_WITH_EXTRACTED_TEXT = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}  # Use extracted text file
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?",
        "expected_answer": "he company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic and regulatory pressures....",       
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?",
        "expected_answer": "The MD&A states that Apple’s 2025 product gross margin was negatively impacted by tariff costs, which partially offset favorable costs and product-mix effects....",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?",
        "expected_answer": "The Management’s Discussion and Analysis document explains that Apple’s 2025 products gross margin decreased in part due to tariff costs, noting that “Products gross margin percentage decreased during 2025 compared to 2024 primarily due to … tariff costs”.....",
        "documents": [
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
]

# Questions comparing TWO documents (2019 vs 2025)
QUESTIONS_MULTI_DOC = [
    {
        "question": "Compare Apple's net income between 2019 and 2025. What was the growth rate?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2019.txt"},
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
    {
        "question": "How did Apple's revenue growth strategy change between 2019 and 2025?",
        "documents": [
            {"type": "txt", "path": "apple_10k_2019.txt"},
            {"type": "txt", "path": "apple_10k_2025.txt"}
        ]
    },
]

#FinQA example, Value Investment
QUESTIONS_WITH_TABLE = [
    {
        "question": """
        
        ### Passage: the following table presents for Apple Inc. (AAPL) key financial metrics over the past decade, highlighting annual averages, maximums, and minimums for various performance indicators from 2016 to 2025.
            Years Ended December 31 (2016–2025) | Annual Average | Maximum | Minimum
            Revenue | 319,578 | 416,161 | 215,639
            Revenue Growth | 6.5% | 33.3% | -7.7%
            Gross Profit | 134,628 | 195,201 | 84,263
            Gross Margin % | 41.2% | 46.9% | 37.8%
            Operating Profit | 92,544 | 133,050 | 60,024
            Operating Margin % | 28.3% | 32.0% | 24.1%
            Earnings Per Share | $4.50 | $7.46 | $2.08
            EPS Growth | 14.3% | 71.0% | -9.8%
            Dividends Per Share | $0.81 | $1.02 | $0.55
            Dividend Growth | 7.6% | 13.3% | 4.1%
            Return on Assets | 21.9% | 30.9% | 13.9%
            Return on Equity | 107.1% | 175.5% | 36.9%
            Return on Invested Capital | 40.4% | 66.7% | 20.8%

            Category | Metric | Value
            Valuation Ratios | P/E | 36.3
            Valuation Ratios | P/B | 55.2
            Valuation Ratios | P/S | 9.8
            Valuation Ratios | EV/S | 9.8
            Valuation Ratios | EV/EBITDA | 28.3
            Valuation Ratios | EV/EBIT | 30.8
            Valuation Ratios | EV/Pretax | 30.8
            Valuation Ratios | EV/FCF | 41.4
            10-Yr Median Returns | ROA | 21.7%
            10-Yr Median Returns | ROE | 110.6%
            10-Yr Median Returns | ROIC | 39.6%
            10-Year CAGR | Revenue | 5.9%
            10-Year CAGR | Assets | 2.2%
            10-Year CAGR | FCF | 3.5%
            10-Year CAGR | EPS | 12.5%
            10-Yr Median Margins | Gross Profit | 40.4%
            10-Yr Median Margins | EBIT | 28.8%
            10-Yr Median Margins | Pre-Tax Income | 29.1%
            Capital Structure (Median) | Assets / Equity | 4.9
            Capital Structure (Median) | Debt / Equity | 1.6
            Capital Structure (Median) | Debt / Assets | 0.3

            ###Question: What is the five-year price-to-earnings (PE) ratio, and is it less than 22.5?
            """,

        "expected_answer": "Current market cap / total earnings (net income) over the past five years, P/E = 35.6 > 22.5, pass",
    },

    {
        "question": """
        
        ### Passage: the following table presents for Apple Inc. (AAPL) key financial metrics over the past decade, highlighting annual averages, maximums, and minimums for various performance indicators from 2016 to 2025.
            Years Ended December 31 (2016–2025) | Annual Average | Maximum | Minimum
            Revenue | 319,578 | 416,161 | 215,639
            Revenue Growth | 6.5% | 33.3% | -7.7%
            Gross Profit | 134,628 | 195,201 | 84,263
            Gross Margin % | 41.2% | 46.9% | 37.8%
            Operating Profit | 92,544 | 133,050 | 60,024
            Operating Margin % | 28.3% | 32.0% | 24.1%
            Earnings Per Share | $4.50 | $7.46 | $2.08
            EPS Growth | 14.3% | 71.0% | -9.8%
            Dividends Per Share | $0.81 | $1.02 | $0.55
            Dividend Growth | 7.6% | 13.3% | 4.1%
            Return on Assets | 21.9% | 30.9% | 13.9%
            Return on Equity | 107.1% | 175.5% | 36.9%
            Return on Invested Capital | 40.4% | 66.7% | 20.8%

            Category | Metric | Value
            Valuation Ratios | P/E | 36.3
            Valuation Ratios | P/B | 55.2
            Valuation Ratios | P/S | 9.8
            Valuation Ratios | EV/S | 9.8
            Valuation Ratios | EV/EBITDA | 28.3
            Valuation Ratios | EV/EBIT | 30.8
            Valuation Ratios | EV/Pretax | 30.8
            Valuation Ratios | EV/FCF | 41.4
            10-Yr Median Returns | ROA | 21.7%
            10-Yr Median Returns | ROE | 110.6%
            10-Yr Median Returns | ROIC | 39.6%
            10-Year CAGR | Revenue | 5.9%
            10-Year CAGR | Assets | 2.2%
            10-Year CAGR | FCF | 3.5%
            10-Year CAGR | EPS | 12.5%
            10-Yr Median Margins | Gross Profit | 40.4%
            10-Yr Median Margins | EBIT | 28.8%
            10-Yr Median Margins | Pre-Tax Income | 29.1%
            Capital Structure (Median) | Assets / Equity | 4.9
            Capital Structure (Median) | Debt / Equity | 1.6
            Capital Structure (Median) | Debt / Assets | 0.3

            ###Question: What is the Five-Year Return on Invested Capital (ROIC)?
            """,

        "expected_answer": "ROIC = 39.6%, last five years’ free cash flow and divide it by the company’s total equity and debt",
    },

    {
        "question": """
        
        ### Passage: the following table presents for Apple Inc. (AAPL) key financial metrics over the past decade, highlighting annual averages, maximums, and minimums for various performance indicators from 2016 to 2025.
            Years Ended December 31 (2016–2025) | Annual Average | Maximum | Minimum
            Revenue | 319,578 | 416,161 | 215,639
            Revenue Growth | 6.5% | 33.3% | -7.7%
            Gross Profit | 134,628 | 195,201 | 84,263
            Gross Margin % | 41.2% | 46.9% | 37.8%
            Operating Profit | 92,544 | 133,050 | 60,024
            Operating Margin % | 28.3% | 32.0% | 24.1%
            Earnings Per Share | $4.50 | $7.46 | $2.08
            EPS Growth | 14.3% | 71.0% | -9.8%
            Dividends Per Share | $0.81 | $1.02 | $0.55
            Dividend Growth | 7.6% | 13.3% | 4.1%
            Return on Assets | 21.9% | 30.9% | 13.9%
            Return on Equity | 107.1% | 175.5% | 36.9%
            Return on Invested Capital | 40.4% | 66.7% | 20.8%

            Category | Metric | Value
            Valuation Ratios | P/E | 36.3
            Valuation Ratios | P/B | 55.2
            Valuation Ratios | P/S | 9.8
            Valuation Ratios | EV/S | 9.8
            Valuation Ratios | EV/EBITDA | 28.3
            Valuation Ratios | EV/EBIT | 30.8
            Valuation Ratios | EV/Pretax | 30.8
            Valuation Ratios | EV/FCF | 41.4
            10-Yr Median Returns | ROA | 21.7%
            10-Yr Median Returns | ROE | 110.6%
            10-Yr Median Returns | ROIC | 39.6%
            10-Year CAGR | Revenue | 5.9%
            10-Year CAGR | Assets | 2.2%
            10-Year CAGR | FCF | 3.5%
            10-Year CAGR | EPS | 12.5%
            10-Yr Median Margins | Gross Profit | 40.4%
            10-Yr Median Margins | EBIT | 28.8%
            10-Yr Median Margins | Pre-Tax Income | 29.1%
            Capital Structure (Median) | Assets / Equity | 4.9
            Capital Structure (Median) | Debt / Equity | 1.6
            Capital Structure (Median) | Debt / Assets | 0.3

            ###Question: Is the revenue growth over last year greater than 5 years ago?
            """,

        "expected_answer": "6.4% > 5.5% (2020), but 6.4% << 33.3% (2021), mild pass",
    },
]

#Additional prompting to avoid coding outputs
QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_1 = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?\nFirst restate the question in your own words.",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?\nFirst restate the question in your own words.",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?\nFirst restate the question in your own words.",
        "expected_answer": "Apple’s 2025 disclosures emphasize that tariffs, macroeconomic volatility, and supply-chain concentration collectively pressure margins and elevate operational risk exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?\nFirst restate the question in your own words.",
        "expected_answer": "The company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic, regulatory, and geopolitical pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?\nFirst restate the question in your own words.",
        "expected_answer": "The MD&A explains that tariff costs negatively affected product gross margins and that future changes in trade policy, supplier concentration, and geopolitical conditions could further pressure margins...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?\nFirst restate the question in your own words.",
        "expected_answer": "Apple states that tariff-related costs reduced product gross margins in 2025 and that continued trade restrictions, retaliatory tariffs, and supply-chain disruptions remain ongoing risks...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    }
]

QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_2 = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "Apple’s 2025 disclosures emphasize that tariffs, macroeconomic volatility, and supply-chain concentration collectively pressure margins and elevate operational risk exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "The company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic, regulatory, and geopolitical pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "The MD&A explains that tariff costs negatively affected product gross margins and that future changes in trade policy, supplier concentration, and geopolitical conditions could further pressure margins...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?\nExplicitly decide whether the task requires calculations, and if not, avoid producing any code in your answer.",
        "expected_answer": "Apple states that tariff-related costs reduced product gross margins in 2025 and that continued trade restrictions, retaliatory tariffs, and supply-chain disruptions remain ongoing risks...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    }
]

QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_3 = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "Apple’s 2025 disclosures emphasize that tariffs, macroeconomic volatility, and supply-chain concentration collectively pressure margins and elevate operational risk exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "The company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic, regulatory, and geopolitical pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "The MD&A explains that tariff costs negatively affected product gross margins and that future changes in trade policy, supplier concentration, and geopolitical conditions could further pressure margins...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?\nAnswer only with narrative text, avoiding any code or calculations.",
        "expected_answer": "Apple states that tariff-related costs reduced product gross margins in 2025 and that continued trade restrictions, retaliatory tariffs, and supply-chain disruptions remain ongoing risks...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    }
]

QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_4 = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?\nDo not complete any computation before answering.",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?\nDo not complete any computation before answering.",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?\nDo not complete any computation before answering.",
        "expected_answer": "Apple’s 2025 disclosures emphasize that tariffs, macroeconomic volatility, and supply-chain concentration collectively pressure margins and elevate operational risk exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?\nDo not complete any computation before answering.",
        "expected_answer": "The company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic, regulatory, and geopolitical pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?\nDo not complete any computation before answering.",
        "expected_answer": "The MD&A explains that tariff costs negatively affected product gross margins and that future changes in trade policy, supplier concentration, and geopolitical conditions could further pressure margins...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?\nDo not complete any computation before answering.",
        "expected_answer": "Apple states that tariff-related costs reduced product gross margins in 2025 and that continued trade restrictions, retaliatory tariffs, and supply-chain disruptions remain ongoing risks...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    }
]

QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_5 = [
    {
        "question": "What is the condensed, overarching explanation of how Apple's global supply-chain dependencies, exposure to geopolitical and macroeconomic pressures, and vulnerability to cybersecurity and data-security threats collectively form interconnected risks that can materially affect its operations, financial condition, and reputation?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "Apple faces an interlinked network of risks stemming from three major areas: (1) its heavily internationalized supply chain, which depends on single-source and geographically concentrated manufacturers; (2) macroeconomic and geopolitical pressures—including tariffs, trade disputes, natural disasters, and public-health disruptions—that can impair production, distribution, consumer demand, and partner stability; and (3) persistent cybersecurity and data-security threats that can disrupt operations, compromise sensitive information, damage customer trust, and trigger regulatory or legal exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the combination of Apple's 2025 product-release cycle, regional shifts in segment performance, and evolving macroeconomic and tariff conditions collectively shape the company's overall changes in net sales, gross margins, and operating expenses during the fiscal year?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "Apple’s 2025 results reflected the interplay of new hardware and software launches, varied regional demand patterns, and significant macroeconomic and tariff pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How do Apple's 2025 financial disclosures characterize the combined impact of new U.S.-announced tariffs, global macroeconomic pressures, and supply-chain dependencies on its gross margins and overall operational risks?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "Apple’s 2025 disclosures emphasize that tariffs, macroeconomic volatility, and supply-chain concentration collectively pressure margins and elevate operational risk exposure...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How would you summarize the company's recent financial performance, operating trends, and major risk exposures based on its latest annual disclosures?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "The company’s recent annual filings describe a year marked by solid operational performance alongside increasing macroeconomic, regulatory, and geopolitical pressures...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "Based on Apple's 2025 Form 10-K, how did newly introduced U.S. tariffs beginning in Q2 2025 affect Apple's product gross margins, and what related supply-chain and international trade risks does the company identify that could further impact margins in the future?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "The MD&A explains that tariff costs negatively affected product gross margins and that future changes in trade policy, supplier concentration, and geopolitical conditions could further pressure margins...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    },
    {
        "question": "How did the new U.S. tariffs introduced in the second quarter of 2025 affect Apple's 2025 product gross margins, and what risks related to these tariffs does Apple identify that could further impact its future financial performance?\nDo not answer with any python code snippets, only provide a natural language analysis.",
        "expected_answer": "Apple states that tariff-related costs reduced product gross margins in 2025 and that continued trade restrictions, retaliatory tariffs, and supply-chain disruptions remain ongoing risks...",
        "documents": [
            { "type": "txt", "path": "apple_10k_2025.txt" }
        ]
    }
]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="queries.json")
    parser.add_argument("--mode", type=str, default="extracted_text",
                        choices=["no_context", "extracted_text", "prompted_extracted_text_1", "prompted_extracted_text_2", "prompted_extracted_text_3", "prompted_extracted_text_4", "prompted_extracted_text_5", "multi_doc", "pdf", "table"],
                        help="Which question set to use")
    args = parser.parse_args()
    
    if args.mode == "no_context":
        questions = QUESTIONS_NO_CONTEXT
    elif args.mode == "extracted_text":
        questions = QUESTIONS_WITH_EXTRACTED_TEXT
    elif args.mode == "prompted_extracted_text_1":
        questions = QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_1
    elif args.mode == "prompted_extracted_text_2":
        questions = QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_2
    elif args.mode == "prompted_extracted_text_3":        
        questions = QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_3
    elif args.mode == "prompted_extracted_text_4":
        questions = QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_4
    elif args.mode == "prompted_extracted_text_5":
        questions = QUESTIONS_WITH_PROMPTED_EXTRACTED_TEXT_5
    elif args.mode == "multi_doc":
        questions = QUESTIONS_MULTI_DOC
    elif args.mode == "table":
        questions = QUESTIONS_WITH_TABLE
    
    create_queries_json(args.output, questions)
    
    print("\nNext step: Run KL divergence computation with:")
    print(f"  python compute_kl_divergence.py --queries_json {args.output}")