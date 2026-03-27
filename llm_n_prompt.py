import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

class PromptTemplate:

    def build_prompt(self, question, contexts):

        context_text = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else doc for doc in contexts
        )

        system = """You are ChefBot, an expert assistant specialising in South Asian cuisine.
You have deep knowledge of dishes, ingredients, spices, cooking techniques, and culinary traditions from India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan, and the Maldives.

Rules you must always follow:
- Answer using ONLY the information in the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say exactly: "I don't have information about that in my knowledge base."
- Never guess, never invent ingredients, dishes, or techniques.
- Be clear, factual, and concise.
- Do not repeat the same point twice.
- When relevant, mention specific ingredients, spices, regional variations, or cooking techniques from the context.
- If the question asks about a dish, include how it is made, key ingredients, and regional variations if the context mentions them.
- If the question asks about a spice or ingredient, explain what it is, how it is used, and which dishes it appears in based on the context."""

        user = f"""CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def build_few_shot_prompt(self, question, contexts):
        context_text = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else doc for doc in contexts
        )

        system = """You are ChefBot, an expert assistant specialising in South Asian cuisine.
    Rules:
    - Answer using ONLY the provided context.
    - If info is missing, say: "I don't have information about that in my knowledge base."
    - Be factual and concise."""

        # These are the "Shots" (Examples)
        few_shot_examples = """
    EXAMPLE 1:
    CONTEXT: Chana Masala is a North Indian chickpea curry. It uses garam masala, amchoor (mango powder), and green chillies.
    QUESTION: What gives Chana Masala its sourness?
    ANSWER: According to the context, Chana Masala gets its sourness from amchoor (mango powder).

    EXAMPLE 2:
    CONTEXT: Traditional Jalebi is made by deep-frying maida flour batter in pretzel shapes and soaking them in sugar syrup.
    QUESTION: Is Jalebi baked?
    ANSWER: No. Based on the information provided, Jalebi is made by deep-frying maida flour batter and then soaking it in sugar syrup.

    EXAMPLE 3:
    CONTEXT: Lassi is a yogurt-based drink from the Punjab region.
    QUESTION: How do you make a Pizza?
    ANSWER: I don't have information about that in my knowledge base.
    """

        user = f"""{few_shot_examples}

    ---
    CURRENT TASK:
    CONTEXT:
    {context_text}

    QUESTION: {question}

    ANSWER:"""

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

# class QwenLLM:
#
#     def __init__(self, device="cpu"):
#
#         print("Loading LLM...")
#
#         self.device = device
#
#         self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#
#         self.model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME
#         )
#
#         self.model.to(self.device)
#
#         self.model.eval()
#
#
#     def generate(self, messages, max_tokens=512):
#
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
#         input_length = inputs.input_ids.shape[1]
#
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 temperature=0.4,
#                 # repetition_penalty=1.3,
#                 do_sample=True,
#             )
#
#         response = self.tokenizer.decode(
#             outputs[0][input_length:],
#             skip_special_tokens=True,
#         )
#
#         return response.strip()
class QwenLLM:

    def __init__(self, device="cpu"):

        print("Loading LLM...")

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if device in ("mps","cuda") else torch.float32
        )

        self.model.to(self.device)

        self.model.eval()


    def generate(self, messages, max_tokens=512):

        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )

        # inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs = self.tokenizer.apply_chat_template(messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.4,
                # repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return response.strip()