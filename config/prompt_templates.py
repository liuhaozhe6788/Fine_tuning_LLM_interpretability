MODEL_ID_TO_TEMPLATES_DICT = {
    "unsloth/mistral-7b-v0.2-bnb-4bit": ("<s>[INST] {}{} [/INST]{}", "[/INST]"),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": ("<s>[INST] {}{} [/INST]{}", "[/INST]"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("<s>[INST] {}{} [/INST]{}", "[/INST]"),
    "microsoft/Orca-2-7b": ("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant{}", "<|im_start|>assistant"),
    "microsoft/Orca-2-13b": ("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant{}", "<|im_start|>assistant"),
    "microsoft/Phi-3-mini-128k-instruct": ("<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>{}", "<|assistant|>"),
    "microsoft/Phi-3-medium-128k-instruct": ("<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>{}", "<|assistant|>")
}

four_shot_prompt_finqa_templates = [
    """Read the following passage and then write python code to answer the question
###Passage: ( in millions ) | dec 282013 | dec 292012
available-for-sale investments | $ 18086 | $ 14001
cash | 854 | 593
equity method investments | 1038 | 992
loans receivable | 1072 | 979
non-marketable cost method investments | 1270 | 1202
reverse repurchase agreements | 800 | 2850
trading assets | 8441 | 5685
total cash and investments | $ 31561 | $ 26302
###Question: what percentage of total cash and investments as of dec . 29 2012 was comprised of available-for-sale investments?
###Python
#Calculate: percent_available_for_sale_investments_dec_29_2012 = available_for_sale_investments_dec_29_2012 / total_cash_and_investments_dec_29_2012
available_for_sale_investments_dec_29_2012 = 14001
total_cash_and_investments_dec_29_2012 = 26302
percent_available_for_sale_investments_dec_29_2012 = available_for_sale_investments_dec_29_2012 / total_cash_and_investments_dec_29_2012
ans = percent_available_for_sale_investments_dec_29_2012
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: the chart shows that the firm posted market risk 2013 related gains on 248 out of 261 days in this period , with 12 days exceeding $ 210 million . december 31 ( in millions ) | 1 basis point increase in jpmorgan chase 2019s credit spread
2010 | $ 35
2009 | $ 39
###Question: on what percent of trading days were there market gains above $ 210 million?
###Python
#Calculate: percent_days_with_market_gains_above_210_million = days_with_market_gains_above_210_million / total_trading_days
days_with_market_gains_above_210_million = 12
total_trading_days = 261
percent_days_with_market_gains_above_210_million = days_with_market_gains_above_210_million / total_trading_days
ans = percent_days_with_market_gains_above_210_million
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $ 75.0 million and network location intangibles of approximately $ 72.7 million . the customer-related intangibles and
network location intangibles are being amortized on a straight-line basis over periods of up to 20 years.
- | preliminary purchase price allocation
current assets | $ 8763
. . .
fair value of net assets acquired | $ 57536
goodwill ( 2 ) | 5998
###Question: for acquired customer-related and network location intangibles , what is the expected annual amortization expenses , in millions?
###Python
#Calculate: amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: the aggregate commitment under the liquidity asset purchase agreements was approximately $ 23.59 billion and $ 28.37 billion at
december 31 , 2008 and 2007 , respectively .
( dollars in billions ) | 2008 amount | 2008 percent of total conduit assets | 2008 amount | percent of total conduit assets
united states | $ 11.09 | 46% ( 46 % ) | $ 12.14 | 42% ( 42 % )
australia | 4.30 | 17 | 6.10 | 21
. . .
greece | 0.27 | 1 | 0.31 | 1
other | 1.01 | 5 | 1.26 | 5
total conduit assets | $ 23.89 | 100% ( 100% ) | $ 28.76 | 100% ( 100% )
###Question: what is percentage change in total conduit asset from 2007 to 2008?
###Python
#Calculate: percent_change_in_total_conduit_assets = ( total_conduit_assets_2008 - total_conduit_assets_2007 ) / total_conduit_assets_2007
total_conduit_assets_2007 = 28.76
total_conduit_assets_2008 = 23.89
net_change_in_total_conduit_assets = total_conduit_assets_2008 - total_conduit_assets_2007
percent_change_in_total_conduit_assets = net_change_in_total_conduit_assets / total_conduit_assets_2007
ans = percent_change_in_total_conduit_assets
###End Python</s>
    """
]

four_shot_prompt_convfinqa_templates = [
    """Read the following text and table, and then answer the last question in a series of questions:
###Passage:
- | shares available for awards | shares subject to outstanding awards
2009 global incentive plan | 2322450 | 2530454
2004 stock incentive plan | - | 5923147
###Questions: how many shares are subject to outstanding awards is under the 2009 global incentive plan? what about under the 2004 stock
incentive plan? how many total shares are subject to outstanding awards? what about under the 2004 stock incentive plan?
###Last Question: what proportion does this represent?
###Python
# Calculate: shares_outstanding_awards_2009_global_incentive_plan / ( shares_outstanding_awards_2009_global_incentive_plan +
shares_outstanding_awards_2004_stock_incentive_plan )
shares_outstanding_awards_2009_global_incentive_plan = 2530454
shares_outstanding_awards_2004_stock_incentive_plan = 5923147
total_shares_outstanding_awards = shares_outstanding_awards_2009_global_incentive_plan +
shares_outstanding_awards_2004_stock_incentive_plan
proportion = shares_outstanding_awards_2009_global_incentive_plan / total_shares_outstanding_awards
ans = proportion
###End Python</s>
    """,
    """Read the following text and table, and then answer the last question in a series of questions:
###Passage: compensation expense the company recorded $ 43 million , $ 34 million , and $ 44 million of expense related to stock awards
for the years ended december 31 , 2015 , 2014 , and 2013 , respectively .
###Questions: what is the compensation expense the company recorded in 2015? what about in 2014? what is the total compensation expense the company recorded in 2015 and 2014? what is the total expenses including 2013?
###Last Question: what is the average for three years?
###Python
# Calculate: average_for_three_years = ( compensation_expense_2015 + compensation_expense_2014 + compensation_expense_2013 ) / 3
compensation_expense_2015 = 43
compensation_expense_2014 = 34
compensation_expense_2013 = 44
total_compensation_expense = compensation_expense_2015 + compensation_expense_2014 + compensation_expense_2013
average_for_three_years = total_compensation_expense / 3
ans = average_for_three_years
###End Python</s>
    """,
    """Read the following text and table, and then answer the last question in a series of questions:
###Passage: the net loss on disposal of those assets was $ 344000 for 2005 and $ 43000 for 2004 .
###Questions: what was the net loss on disposal of assets in 2005? what was the value in 2004? what was the change in value?
###Last Question: what was the percent change?
###Python
# Calculate: percent_change = ( net_loss_on_disposal_of_assets_2005 - net_loss_on_disposal_of_assets_2004 ) /
net_loss_on_disposal_of_assets_2004
net_loss_on_disposal_of_assets_2005 = 344000
net_loss_on_disposal_of_assets_2004 = 43000
net_change_in_value = net_loss_on_disposal_of_assets_2005 - net_loss_on_disposal_of_assets_2004
percent_change = net_change_in_value / net_loss_on_disposal_of_assets_2004
ans = percent_change
###End Python</s>
    """,
    """Read the following text and table, and then answer the last question in a series of questions:
###Passage: location | operations conducted | approximatesquare feet | leaseexpirationdates
dublin ireland | global supply chain distribution and administration offices | 160000 | owned
athlone ireland | commercial research and development manufacturing | 80000 | owned
bogart georgia | commercial research and development manufacturing | 70000 | owned
smithfield rhode island | commercial research and development manufacturing | 67000 | owned
###Questions: what is the square feet of the owned global supply chain distribution and administration offices? what is the square feet of the
owned commercial research and development manufacturing? what is the sum of those values? what is the total sum including square feet of
commercial research and development manufacturing in bogart, georgia? what is the total sum including square feet of commercial research
and development manufacturing in smithfield, rhode island?
###Last Question: what is the total sum of square feet owned?
###Python
# Calculate: owned_global_supply_chain_distribution_dublin + commercial_research_and_development_manufacturing_athlone + commercial_research_and_development_manufacturing_bogart + commercial_research_and_development_manufacturing_smithfield
owned_global_supply_chain_distribution_dublin = 160000
commercial_research_and_development_manufacturing_athlone = 80000
commercial_research_and_development_manufacturing_bogart = 70000
commercial_research_and_development_manufacturing_smithfield = 67000
total_square_feet_owned = owned_global_supply_chain_distribution_dublin + commercial_research_and_development_manufacturing_athlone +
commercial_research_and_development_manufacturing_bogart + commercial_research_and_development_manufacturing_smithfield
ans = total_square_feet_owned
###End Python</s>
    """
]

four_shot_prompt_tatqa_templates = [
    """Read the following passage and then write python code to answer the question
###Passage: 17. Income Taxes
Income before income taxes for the Company’s domestic and foreign operations was as follows:
— | — | Years Ended June 30, | —
($ in millions) | 2019 | 2018 | 2017
Domestic | $204.2 | $140.3 | $56.0
Foreign | 11.8 | 19.9 | 14.2
Income before income taxes | $216.0 | $160.2 | $70.2
###Question: What was the change in Foreign in 2019 from 2018?
###Python
# Calculate: change_in_foreign = foreign_in_2019 - foreign_in_2018
foreign_in_2018 = 19.9
foreign_in_2019 = 11.8
ans = foreign_in_2019 - foreign_in_2018
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: 11 Intangible assets (continued) (a) Intangible assets RIGHTS AND LICENCES Certain licences that NEXTDC possesses have
an indefinite useful life and are carried at cost less impairment losses and are subject to impairment review at least annually and whenever
there is an indication that it may be impaired.
. . .
— | Rights and licenses | Internally generated software | Software under development | Total
At 30 June 2019 | — | — | — | —
Cost | 13 | 12,961 | 16,284 | 29,259
Accumulated amortisation | - | -5,580 | - | -5,580
At 30 June 2018 | — | — | — | —
Cost | 104 | 9,555 | 6,509 | 16,168
Accumulated amortisation | -91 | -3,170 | - | -3,261
Net book amount | 13 | 6,385 | 6,509 | 12,907
###Question: Which year have greater total accumulated amortisation?
###Python
# Calculate: find total accumulated amortization and choose year with maximum accumulated amortisation
total_accumulated_amortisation = {’2019’: 5580, ’2018’: 3261}
ans = sorted(total_accumulated_amortisation.items(), key=lambda tup: tup[1], reverse=True)[0][0]
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: The following table sets forth the breakdown of revenues by category and segment.
. . .
Year Ended December 31, | — | —
— | 2019 | 2018
Total Asia Pacific revenues | 6,490 | 7,859
Total Europe revenues | 36,898 | 36,149
Total North America revenues | 68,024 | 67,314
Total revenues | $111,412 | 111,322
###Question: In 2019, how many geographic regions have total revenues of more than $20,000 thousand?
###Python
# Calculate: find locations with revenue more than $20,000 in a list and count
total_revenues_in_all_regions = {’Asia Pacific’: 6490, ’Europe’: 36898, ’North America’: 68024}
regions_have_more_than_20000_thousand_total_revenues = [k for k, v in total_revenues_in_all_regions.items() if v > 20000]
ans = len(regions_have_more_than_20000_thousand_total_revenues)
###End Python</s>
    """,
    """Read the following passage and then write python code to answer the question
###Passage: Effective Income Tax Rate A reconciliation of the United States federal statutory income tax rate to our effective income tax rate
is as follows: In 2019 and 2018 we had pre-tax losses of $19,573 and $25,403, respectively, which are available for carry forward to offset
future taxable income.
— | Year Ended | Year Ended
— | December 31, 2018 | December 31, 2019
United States federal statutory rate | 21.00% | 21.00%
State taxes, net of federal benefit | 1.99% | -0.01%
###Question: What was the 2019 percentage change in pre-tax losses?
###Python
# Calculate: percentage_change_in_pre_tax_losses = ( pre_tax_losses_2019 - pre_tax_losses_2018 ) / pre_tax_losses_2018 * 100
pre_tax_losses_2018 = 25403
pre_tax_losses_2019 = 19573
net_change = pre_tax_losses_2019 - pre_tax_losses_2018
ans = net_change / pre_tax_losses_2018 * 100
###End Python</s>
    """
]

zero_shot_eval_instruction = """First, identify entities required to answer the
    question. Extract the identified entities and store in python variables. Then perform calculations with the entities and strictly
    store the answer to the python variable "ans" in one value. Python code must
    end after the variable "ans" is defined. Comments must begin
    with character "#". Python code must be in one section enclosed in ```python\n your code here ```."""

few_shot_eval_instruction = """Python code must
    end after the variable "ans" is defined. Comments must begin
    with character "#"."""

def build_teacher_prompt(context: str, question: str, program: str) -> str:
    """
    Build the prompt template for code generation with the teacher model.
    
    Args:
        context: The document/table context for the current question
        question: The question to answer
        program: The program to use for code generation
        
    Returns:
        Formatted prompt string for the teacher model
    """
    return f"""Read the following passage and then write python code to answer the question:
    ###Passage: {context}
    ###Question: {question}
    ###Answer Hint: Strictly perform the following calculations to arrive at the answer::{program}
    ###Python
    """

def build_teacher_prompt_with_conversation_history(context: str, question: str, program: str, conversation_history: str) -> str:
    """
    Build the prompt template for code generation with the teacher model that contains previous questions.
    
    Args:
        context: The document/table context for the current question
        question: The question to answer
        program: The program to use for code generation
        conversation_history: The previous questions in conversation
        
    Returns:
        Formatted prompt string for the teacher model
    """
    return f"""Read the following passage and then write python code to answer the question:
    ###Passage: {context}
    ###Questions: {conversation_history}
    ###Last Question: {question}
    ###Answer Hint: Strictly perform the following calculations to arrive at the answer::{program}
    ###Python
    """


def build_finetuning_prompt(context: str, question: str) -> str:
    """
    Build the prompt template for fine-tuning (combines context and question).
    
    Args:
        context: The document/table context for the current question
        question: The question to answer
        
    Returns:
        Formatted prompt string for fine-tuning
    """
    return f"""Read the following passage and then write python code to answer the question:   
    ###Passage: {context}
    ###Question: {question}
    ###Python
    """

def build_finetuning_prompt_with_conversation_history(context: str, question: str, conversation_history: str) -> str:
    """
    Build the prompt template for fine-tuning (combines context and question) that contains previous questions.
    
    Args:
        context: The document/table context for the current question
        question: The question to answer
        conversation_history: The previous conversation history
    Returns:
        Formatted prompt string for fine-tuning
    """
    return f"""Read the following text and table, and then answer the last question by writing a python code:   
    ###Passage: {context}
    ###Questions: {conversation_history}
    ###Last Question: {question}
    ###Python
    """
    
