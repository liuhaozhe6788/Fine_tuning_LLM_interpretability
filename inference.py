import torch, os
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

#use bf16 and FlashAttention if supported
compute_dtype = torch.float16
attn_implementation = 'sdpa'

adapter= "data/models/FinQA/mistralai_Mistral-7B-Instruct-v0.3-peftq_proj_k_proj_v_proj_o_proj-bs1/model"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=compute_dtype,
    device_map={"": 0},
    attn_implementation=attn_implementation,
)

model = PeftModel.from_pretrained(model, adapter)

prompt = "<s>[INST] {}{} [/INST]".format("Read the following passage and then write python code to answer the question:   ", """    ###Passage: part ii item 5 .
market for registrant 2019s common equity , related stockholder matters and issuer purchases of equity securities the following table presents reported quarterly high and low per share sale prices of our common stock on the new york stock exchange ( 201cnyse 201d ) for the years 2010 and 2009. .

2010|high|low
quarter ended march 31|$ 44.61|$ 40.10
quarter ended june 30|45.33|38.86
quarter ended september 30|52.11|43.70
quarter ended december 31|53.14|49.61
2009|high|low
quarter ended march 31|$ 32.53|$ 25.45
quarter ended june 30|34.52|27.93
quarter ended september 30|37.71|29.89
quarter ended december 31|43.84|35.03

on february 11 , 2011 , the closing price of our common stock was $ 56.73 per share as reported on the nyse .
as of february 11 , 2011 , we had 397612895 outstanding shares of common stock and 463 registered holders .
dividends we have not historically paid a dividend on our common stock .
payment of dividends in the future , when , as and if authorized by our board of directors , would depend upon many factors , including our earnings and financial condition , restrictions under applicable law and our current and future loan agreements , our debt service requirements , our capital expenditure requirements and other factors that our board of directors may deem relevant from time to time , including the potential determination to elect reit status .
in addition , the loan agreement for our revolving credit facility and term loan contain covenants that generally restrict our ability to pay dividends unless certain financial covenants are satisfied .
for more information about the restrictions under the loan agreement for the revolving credit facility and term loan , our notes indentures and the loan agreement related to our securitization , see item 7 of this annual report under the caption 201cmanagement 2019s discussion and analysis of financial condition and results of operations 2014liquidity and capital resources 2014factors affecting sources of liquidity 201d and note 6 to our consolidated financial statements included in this annual report. .
    ###Question: what is the average number of shares per registered holder as of february 11 , 2011?""")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, do_sample=False, temperature=0.0, max_new_tokens=150)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
