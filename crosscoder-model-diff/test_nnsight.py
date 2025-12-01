from nnsight import LanguageModel
import torch

# base_model = LanguageModel('mistralai/Mistral-7B-Instruct-v0.3', device_map='cuda:0')
chat_model = LanguageModel('liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora', device_map='cuda:0')

# with base_model.trace(["The Eiffel Tower is in the city of Paris."]) as tracer:
#     base_model_acts = base_model.model.layers[16].mlp.output.save()

for i in range(10):
    with chat_model.trace(["The Eiffel Tower is in the city of Paris."] * 4) as tracer:
        chat_model_acts = chat_model.model.layers[16].mlp.output.save()
    print(chat_model_acts)
# assert not torch.equal(base_model_acts, chat_model_acts)
