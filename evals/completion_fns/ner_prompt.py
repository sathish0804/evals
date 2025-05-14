from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
from openai import OpenAI
import os

# Create a class that inherits from CompletionResult
class NerPromptResult(CompletionResult):
    def __init__(self, completion):
        self.completion = completion

    def get_completions(self) -> list[str]:
        return [self.completion]

class NerPrompt(CompletionFn):
    def __init__(self, model="gpt-4", **kwargs):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open("evals/prompt/ner_prompt.txt", "r") as f:
            self.prompt_template = f.read()

    def __call__(self, prompt_args, **kwargs) -> CompletionResult:
        # Handle different input types for prompt_args
        if isinstance(prompt_args, dict):
            # If it's a dictionary, extract fields normally
            user_query = prompt_args.get("input")
            today_date = prompt_args.get("today_date", "2025-05-13")
            profile_name = prompt_args.get("profile_name", "default_user")
            tenant_id = prompt_args.get("tenant_id", "unknown")
            facility_info = prompt_args.get("facility_info", "[]")
        elif isinstance(prompt_args, list):
            # If it's a list (likely a chat message format), use the prompt directly
            # This happens when NerPrompt is used as the evaluator
            prompt = prompt_args[0]["content"] if prompt_args and isinstance(prompt_args[0], dict) and "content" in prompt_args[0] else ""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt_args,
                temperature=0
            )
            completion = response.choices[0].message.content.strip()
            record_sampling(prompt=str(prompt_args), sampled=completion)
            return NerPromptResult(completion)
        else:
            # If it's a string or other type, use it directly as the user query
            user_query = str(prompt_args)
            today_date = "2025-05-13"
            profile_name = "default_user"
            tenant_id = "unknown"
            facility_info = "[]"

        # Format the prompt with the extracted fields
        prompt = self.prompt_template.format(
            today_date=today_date,
            user_message=user_query,
            facility_info=facility_info,
            tenant_id=tenant_id,
            profile_name=profile_name
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        completion = response.choices[0].message.content.strip()
        record_sampling(prompt=prompt, sampled=completion)
        return NerPromptResult(completion)
