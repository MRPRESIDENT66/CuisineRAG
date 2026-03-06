import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class PromptTemplate:

    def build_prompt(self, question, contexts):

        context_text = "\n\n".join(contexts)

        prompt = f"""
You are **ChefBot**, an expert in South Asian cuisine with deep knowledge of dishes, ingredients, cooking techniques, and culinary traditions from:

• India  
• Pakistan  
• Bangladesh  
• Sri Lanka  
• Nepal  
• Bhutan  
• Maldives  

You are answering questions using information retrieved from a **South Asian cuisine knowledge base**.

---------------------
CONTEXT
---------------------
{context_text}
---------------------

INSTRUCTIONS

1. Answer the question using ONLY the information in the provided context.
2. Do NOT invent information or use outside knowledge.
3. If the context does not contain enough information to answer the question, say:
   "The provided context does not contain enough information to answer this question."
4. Be clear, informative, and concise.
5. When relevant, mention ingredients, spices, cooking techniques, or regional variations.

---------------------
QUESTION
---------------------
{question}

---------------------
ANSWER
---------------------
"""

        return prompt


class QwenLLM:

    def __init__(self, device="cpu"):

        print("Loading LLM...")

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)

        self.model.eval()


    def generate(self, prompt, max_tokens=200):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )

        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()