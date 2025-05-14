from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
from openai import OpenAI
import os

# Create a custom CompletionResult implementation
class RulPlannerCompletionResult(CompletionResult):
    def __init__(self, text):
        self.text = text

    def get_completions(self) -> list[str]:
        return [self.text]

class RulPlannerPrompt(CompletionFn):
    def __init__(self, model="gpt-4", **kwargs):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open("evals/prompt/rul_planner.txt", "r") as f:
            self.prompt_template = f.read()

    def __call__(self, prompt=None, prompt_args=None, **kwargs) -> CompletionResult:
        # Handle both types of calls
        if prompt_args is not None:
            # Called with prompt_args parameter
            user_query = prompt_args.get("input")
            today_date = prompt_args.get("today_date", "2025-05-13")
            profile_name = prompt_args.get("profile_name", "default_user")
            tenant_id = prompt_args.get("tenant_id", "unknown")
            facility_info = prompt_args.get("facility_info", "[]")
        elif isinstance(prompt, dict):
            # Called with a dictionary as prompt
            user_query = prompt.get("input")
            today_date = prompt.get("today_date", "2025-05-13")
            profile_name = prompt.get("profile_name", "default_user") 
            tenant_id = prompt.get("tenant_id", "unknown")
            facility_info = prompt.get("facility_info", "[]")
        else:
            # Fallback to treating prompt as the user query
            user_query = prompt
            today_date = "2025-05-13"  # Default values
            profile_name = "default_user"
            tenant_id = "unknown"
            facility_info = "[]"

        prompt = self.prompt_template.format(
            today_date=today_date,
            user_query=user_query,
            facility_info=facility_info,
            tenant_id=tenant_id,
            profile_name=profile_name
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # Get the raw completion from the model
        raw_completion = response.choices[0].message.content.strip()
        
        # Record the completion for logging purposes
        record_sampling(prompt=prompt, sampled=raw_completion)
        
        # Return the raw completion - the evaluator will compare this against the ideal response
        return RulPlannerCompletionResult(raw_completion)
