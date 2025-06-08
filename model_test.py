from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch 

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "PATH_TO_MOUDLE"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if "cuda" in device else "auto", 
    device_map="auto"  
).eval()

while(True):
    prompt = input("请输入问题:")
    if prompt.lower() == "exit":
        print("Exiting...")
        exit()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids

    streamer = TextStreamer(tokenizer)
    outputs = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
