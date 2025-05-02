from transformers import AutoModelForCausalLM, AutoTokenizer


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("Model_Tokenizer/qwen3_0.6_tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "Model_Tokenizer/qwen3_0.6_model",
    torch_dtype="auto",
    device_map="auto"
)

class ChatBot:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

        