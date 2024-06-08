MODEL_NAME = "indischepartij/MiaLatte-Indo-Mistral-7b"
LORA_NAME = "xMaulana/QLoRA-Psychika"

import torch
from transformers import AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from psychika import generate_text_from_chat, PSYCHIKA_BNBCONFIG

device = "cuda" #harus pakai cuda

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MistralForCausalLM.from_pretrained(MODEL_NAME, quantization_config=PSYCHIKA_BNBCONFIG, device_map="auto", low_cpu_mem_usage=True)
lora_model = PeftModel.from_pretrained(
    model,
    LORA_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = lora_model.merge_and_unload().eval()

messages = [{"role": "user", 
             "content": "Aku merasa kurang bersemangat hari ini, apakah kamu bisa membantuku agar semangat?"}]

hasil = generate_text_from_chat(tokenizer, model, messages)

print(hasil)
