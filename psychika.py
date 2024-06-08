import torch
from transformers import LlamaTokenizerFast, MistralForCausalLM, BitsAndBytesConfig

PSYCHIKA_BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant=False
)


def generate_text_from_chat(tokenizer:LlamaTokenizerFast, model:MistralForCausalLM, chat:list, include_prompt=False, partition=True, only_return_token=False, device:str="cuda", temperature:float=0.7, top_p:float=0.90,top_k:int=6):
    for i in chat:
        if not isinstance(i, dict):
            raise Exception(f"{i} bukan merupakan sebuah dictionary")
    

    prompt = [{
              "role": "user",
              "content": """Di bawah ini adalah instruksi yang menjelaskan sebuah task. Tuliskan jawaban yang dapat menjawab permintaan dengan tepat.
              Kamu adalah seorang psikiater digital, namamu adalah Psychika, tugasmu adalah untuk membantu saya untuk dapat mengatasi masalah yang saya miliki terkait masalah mental. 
              """
            },
            {
                "role": "assistant", 
                "content": "Aku adalah Psychika, psikiater pribadi kamu. Saya akan membantu kamu untuk mengatasi masalah mental yang kamu miliki"
                }]
    inputs = tokenizer.apply_chat_template(prompt+chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    output = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, output_scores=True, max_new_tokens=1024, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    
    if not include_prompt:
      dec_out = tokenizer.decode(output[0][len(inputs["input_ids"][0])], skip_special_tokens=True)
    else:
      dec_out = tokenizer.decode(output[0], skip_special_tokens=True)

    if only_return_token:
      return [output[0]]

    if partition:
      return [dec_out.rpartition(".")[0], output[0]]

    return [dec_out, output[0]]



