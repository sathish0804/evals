from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
import openai
from openai import OpenAI

class QnASolverPromptResult(CompletionResult):
    def __init__(self, completion):
        self.completion = completion
        
    def get_completions(self):
        return [self.completion]
        
    def __str__(self):
        return str(self.completion)
        
    def __repr__(self):
        return f"QnASolverPromptResult(completion={repr(self.completion)})"
        
    @property
    def raw_data(self):
        return {'choices': [{'text': self.completion}]}
        
    @property
    def model(self):
        return "QnASolverPrompt"

class QnASolverPrompt(CompletionFn):
    def __init__(self, model="gpt-4", **kwargs):
        self.model = model
        self.client = OpenAI()
        self.prompt_template = """
You are an expert in solar and wind energy operational metrics. Your task is to analyze the given user task and related observations, perform necessary calculations, and provide a concise, one-line solution.
Given the problem below
{problem}
you have come up with the following plan
{plan}
Here are the observations for each step of your plan below
{observations}.            
Now solve the question and respond with the answer directly with 1 or 2 sentences.

Process:
1. Carefully read the user task and all provided observations.
2. Identify the key metrics and variables involved in the task.
3. Determine the appropriate calculations or formulas needed to solve the problem.
4. Perform all required calculations using the formulas provided below.
5. Synthesize the results into a clear, concise conclusion.

Formulas and Calculations:
- Energy Deviation (%) = ((Actual Energy - Expected Energy) / Expected Energy) * 100
- Wind Power Output (W) = 0.5 * Air Density * Swept Area * Wind Speed^3 * Power Coefficient
- Solar Panel Efficiency (%) = (Electrical Power Output / (Solar Irradiance * Panel Area)) * 100
- All performance metrics and power generation values are in MW units

Additional complex calculations may be performed as needed based on the specific task requirements.

Output format:
Provide only the final answer as a single, concise statement. Do not include any explanations, calculations, or observations used to arrive at the solution.

Example task: Calculate the energy deviation for a solar farm given the expected energy of 1000 MWh and actual energy production of 950 MWh.
Example output: The energy deviation for the solar farm is -5%.

Remember to adjust your calculations and output based on the specific metrics and requirements of each task presented.
"""

    def __call__(self, prompt_args=None, **kwargs):
        try:
            # Allow prompt to be passed as a keyword argument
            if prompt_args is None and 'prompt' in kwargs:
                prompt_args = kwargs.pop('prompt')
            
            # Handle different input formats
            if isinstance(prompt_args, dict) and "problem" in prompt_args:
                # Original dictionary format with direct problem key
                problem = prompt_args["problem"]
                plan = "\n".join(prompt_args["plan"]) if isinstance(prompt_args["plan"], list) else prompt_args["plan"]
                observations = "\n".join([f"{k}: {v}" for k, v in prompt_args["observations"].items()]) if isinstance(prompt_args["observations"], dict) else prompt_args["observations"]
            elif isinstance(prompt_args, dict) and "input" in prompt_args and isinstance(prompt_args["input"], dict):
                # Input is in an "input" field
                input_data = prompt_args["input"]
                problem = input_data.get("problem", "")
                plan = "\n".join(input_data.get("plan", [])) if isinstance(input_data.get("plan"), list) else input_data.get("plan", "")
                observations = "\n".join([f"{k}: {v}" for k, v in input_data.get("observations", {}).items()]) if isinstance(input_data.get("observations"), dict) else input_data.get("observations", "")
            else:
                # Try other formats
                try:
                    if isinstance(prompt_args, dict):
                        # Try to extract from any dict structure
                        problem = ""
                        plan = ""
                        observations = ""
                        
                        # Look for problem in common keys
                        for key in ["problem", "question", "task", "input"]:
                            if key in prompt_args and isinstance(prompt_args[key], str):
                                problem = prompt_args[key]
                                break
                                
                        # Extract plan and observations if available
                        if "plan" in prompt_args:
                            if isinstance(prompt_args["plan"], list):
                                plan = "\n".join(prompt_args["plan"])
                            elif isinstance(prompt_args["plan"], str):
                                plan = prompt_args["plan"]
                                
                        if "observations" in prompt_args:
                            if isinstance(prompt_args["observations"], dict):
                                observations = "\n".join([f"{k}: {v}" for k, v in prompt_args["observations"].items()])
                            elif isinstance(prompt_args["observations"], str):
                                observations = prompt_args["observations"]
                    else:
                        # Fallback - treat the entire input as the problem
                        problem = str(prompt_args) if prompt_args is not None else ""
                        plan = ""
                        observations = ""
                except Exception as e:
                    # Last resort fallback
                    problem = str(prompt_args) if prompt_args is not None else ""
                    plan = ""
                    observations = ""

            prompt = self.prompt_template.format(problem=problem, plan=plan, observations=observations)

            try:
                # Try the new OpenAI client format first
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                completion = response.choices[0].message.content.strip()
            except (AttributeError, ImportError, Exception) as e:
                # Fall back to old format if necessary
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                    )
                    completion = response.choices[0].message["content"].strip()
                except Exception as e2:
                    # If both approaches fail, return a generic error message
                    print(f"Error calling OpenAI API: {str(e)}, then {str(e2)}")
                    completion = "Error generating completion. Please check your API keys and permissions."

            record_sampling(prompt=prompt, sampled=completion)
            return QnASolverPromptResult(completion)
            
        except Exception as e:
            # Last resort error handling
            print(f"Unexpected error in QnASolverPrompt.__call__: {str(e)}")
            return QnASolverPromptResult("Error in QnASolverPrompt: " + str(e))
