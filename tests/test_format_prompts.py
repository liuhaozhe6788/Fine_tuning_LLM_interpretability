import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils.utils import format_prompts
from config.prompt_templates import MODEL_ID_TO_TEMPLATES_DICT
from config.generation_config import default_config


def test_format_prompts_mistral_template():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    prompt_template, _ = MODEL_ID_TO_TEMPLATES_DICT[model_id]

    val_prompt = (
        "Read the following passage and then write python code to answer the question\n"
        "###Passage: Some context\n"
        "###Question: What is the value?\n"
        "###Python\n"
    )
    generated_code = "ans = 42\n"

    examples = {
        "prompt": [val_prompt],
        "generated_code": [generated_code],
    }

    formatted = format_prompts(
        examples=examples,
        prompt_template=prompt_template,
    )

    assert len(formatted) == 1
    assert formatted[0].startswith("<s>[INST] ")
    assert formatted[0].endswith(default_config.code_end_marker)

    print(formatted[0])

if __name__ == "__main__":
    test_format_prompts_mistral_template()

