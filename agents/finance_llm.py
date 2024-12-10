from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-chat")
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-chat")

# Put your input here:
user_input = """Analyze the impact of Tesla stocks on election day, considering that Elon Musk has openly declared support for Donald Trump."""

# Apply the prompt template and system prompt of LLaMA-2-Chat demo for chat models (NOTE: NO prompt template is required for base models!)
our_system_prompt = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"  # Please do NOT change this
prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{user_input} [/INST]"

# # NOTE:
# # If you want to apply your own system prompt, please integrate it into the instruction part following our system prompt like this:
# your_system_prompt = "Please, check if the answer can be inferred from the pieces of context provided."
# prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{your_system_prompt}\n{user_input} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
    model.device
)
outputs = model.generate(input_ids=inputs, max_length=4096)[0]

answer_start = int(inputs.shape[-1])
pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)

print(pred)
