<!-- 8079e28b-c8d6-44c9-9f04-3655883da471 52743320-23ef-4f69-afce-88dd2e268d47 -->
# Preprocessing Guide Plan

## Review Data Structure

Inspect sample entries to summarize their schema and notable nested fields.

### FinQA
- Financial QA given text+table

### ConvFinQA
- Financial QA given text+table and 4 turns of conversation context.

### TAT-QA
- Multiple financial QA given text+table

## Detail Step-by-Step Instructions

Provide numbered steps for loading, normalizing, and exporting datasets suitable for Hugging Face fine-tuning, including code snippets or command usage.

### FinQA
- Select pre_text (List[str]), post_text (List[str]), table (List(List[str])), and qa (Dict) columns
- Pretty format the list of texts and the table data
- Extract question/answer/program keys in qa dict
- Remove instances without question/answer

### ConFinQA
- Select pre_text (List[str]), post_text (List[str]), table (List(List[str])), and annotation (Dict) columns from train.json and dev.json
- Pretty format the list of texts and the table data
- Extract previous questions in conversation/last answer/last program in annotation dict
- Remove instances without question/answer
- Split the train set into train set and dev set with the ratio of 9:1. The dev set is used as test set.

### TAT-QA
- Select table, paragraph, and multiple question/answer/derivation with arithmetic type from questions dict.
- Pretty format the list of texts and the table data
- Remove instances without question/answer
- Expand multiple QAs in one data instance into multiple data instances for each QA that share the same context. 
- Split the train set into train set and dev set with the ratio of 9:1. The dev set is used as test set.
