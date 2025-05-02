# import transformers
# from transformers import AutoTokenizer
# import torch
# from torch import cuda, bfloat16

# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=bfloat16
# )

# model_id = "Qwen/Qwen3-0.6B"

# model_config = transformers.AutoConfig.from_pretrained(
#     model_id,
# )
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     config=model_config,
#     quantization_config=bnb_config,
#     device_map='auto',
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

# huggingface-cli login  #  login

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)



#SAVE MODEL AND TOKENIZER
model.save_pretrained("Model_Tokenizer/model", from_pt=True)
tokenizer.save_pretrained("Model_Tokenizer/tokenizer", from_pt=True)