# You need to bypass CUDA checks in transformers by modifying transformers and build from source
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

MAX_NEW_TOKENS = 64
model_name = "facebook/opt-1.3b"
# model_name = "./gpt-j-6b"

text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

print('[info] Loading model...')
t0 = time.time()
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=False,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map='auto',
  torch_dtype=torch.bfloat16,
  quantization_config=nf4_config,
)
print('[info] model loaded, time elapsed', round(time.time() - t0, 3), "sec")
print('[info] model dtype', model.dtype)

with torch.no_grad():
    print('[info] warmup run')
    for _ in range(3):
        model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    print('[info] measure latency')
    t0 = time.time()
    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    latency = time.time() - t0
    result = "| latency: " + str(round(latency * 1000, 3)) + " ms |"
    print('+' + '-' * (len(result) - 2) + '+')
    print(result)
    print('+' + '-' * (len(result) - 2) + '+')
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("output:", output, " (type =", type(output), ", len =", len(output), ")")
    quit()

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=-1))
    # prof.export_chrome_trace("bnb_int8_cpu.json")

with torch.no_grad(), torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(
        wait=0, warmup=3, active=1, repeat=0),
    on_trace_ready=trace_handler
    ) as p:
        for i in range(4):
            generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
            p.step()
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("output:", output, " (type =", type(output), ", len =", len(output), ")")
