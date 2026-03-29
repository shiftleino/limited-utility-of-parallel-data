import asyncio
import json
import argparse
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mistralai import Mistral
from .prompts import COHERENCE_PROMPT_TEMPLATE, FLUENCY_PROMPT_TEMPLATE

load_dotenv()

class EvalResponse(BaseModel):
    answer: int = Field(..., description="The category number that best matches the evaluation of the text.")
    reasoning: str = Field(..., description="A short explanation of the evaluation.")

async def evaluate_coherence_mistral(context: str, continuation: str, client: Mistral, semaphore: asyncio.Semaphore, model_name: str, retries: int = 3) -> EvalResponse:
    """
    Sends a single text to the Mistral API for evaluation.
    """
    async with semaphore:
        prompt = COHERENCE_PROMPT_TEMPLATE.format(context=context, continuation=continuation)
        current_attempt = 0
        while current_attempt < retries:
            try:
                print(f"Processing continuation: \"{continuation[:50]}...\"")
                response = await client.chat.complete_async(
                    model= model_name,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1200
                )
                print(f"Response: {response.choices[0].message.content[:50]}...{response.choices[0].message.content[-20:]}")
                coherence_result = response.choices[0].message.content
                if coherence_result.startswith("```json"):
                    json_content = coherence_result.removeprefix("```json\n").removesuffix("\n```").strip()
                elif coherence_result.startswith("```"):
                    json_content = coherence_result.removeprefix("```\n").removesuffix("\n```").strip()
                else:
                    json_content = coherence_result
                
                data = json.loads(json_content)
                return data

            except Exception as e:
                print(f"API error or unexpected issue for sentence: \"{continuation[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    raise e
                await asyncio.sleep(2**current_attempt)

async def evaluate_fluency_mistral(continuation: str, client: Mistral, semaphore: asyncio.Semaphore, model_name: str, retries: int = 3) -> EvalResponse:
    """
    Sends a single text to the Mistral API for evaluation.
    """
    async with semaphore:
        prompt = FLUENCY_PROMPT_TEMPLATE.format(continuation=continuation)
        current_attempt = 0
        while current_attempt < retries:
            try:
                print(f"Processing continuation: \"{continuation[:50]}...\"")
                response = await client.chat.complete_async(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                print(f"Response: {response.choices[0].message.content[:50]}...{response.choices[0].message.content[-20:]}")
                fluency_result = response.choices[0].message.content
                
                if fluency_result.startswith("```json"):
                    json_content = fluency_result.removeprefix("```json\n").removesuffix("\n```").strip()
                elif fluency_result.startswith("```"):
                    json_content = fluency_result.removeprefix("```\n").removesuffix("\n```").strip()
                else:
                    json_content = fluency_result

                data = json.loads(json_content)
                return data

            except Exception as e:
                print(f"API error or unexpected issue for sentence: \"{continuation[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    raise e
                await asyncio.sleep(2**current_attempt)

async def eval_hyperparam_search(args, client, semaphore):    
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    all_evals = {}
    for scaling_factor, items in data["controlled_results_by_a"].items():
        print(f"Processing scaling factor: {scaling_factor}")
        tasks = []
        
        all_evals[scaling_factor] = []
        for item in items:
            context = item.get("context", "")
            continuation = item.get("completion", "")

            if not context or not continuation:
                raise ValueError("Context and continuation must not be empty.")
            
            result_item = {
                "context": context,
                "continuation": continuation,
            }
            all_evals[scaling_factor].append(result_item)
            
            tasks.append(evaluate_coherence_mistral(context, continuation, client, semaphore, args.model_name))
            tasks.append(evaluate_fluency_mistral(continuation, client, semaphore, args.model_name))
        
        results = await asyncio.gather(*tasks)
        for i in range(len(items)):
            coherence_result = results[i * 2]
            fluency_result = results[i * 2 + 1]
            
            all_evals[scaling_factor][i]["coherence"] = coherence_result["answer"]
            all_evals[scaling_factor][i]["coherence_reasoning"] = coherence_result["reasoning"]
            all_evals[scaling_factor][i]["fluency"] = fluency_result["answer"]
            all_evals[scaling_factor][i]["fluency_reasoning"] = fluency_result["reasoning"]

        print(f"Completed processing for scaling factor: {scaling_factor}")
    
    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(all_evals, f, indent=4)

async def eval_main_generations(args, client, semaphore):
    with open(args.input_file, "r") as f:
        data = json.load(f)

    output = []
    tasks = []
    for item in data:
        context = item.get("context", "")
        continuation = item.get("completion", "")

        if not context or not continuation:
            raise ValueError("Context and continuation must not be empty.")
        
        tasks.append(evaluate_coherence_mistral(context, continuation, client, semaphore, args.model_name))
        tasks.append(evaluate_fluency_mistral(continuation, client, semaphore, args.model_name))
    
    results = await asyncio.gather(*tasks)
    for i in range(len(data)):
        coherence_result = results[i * 2]
        fluency_result = results[i * 2 + 1]
        
        output.append({
            "context": data[i].get("context", ""),
            "completion": data[i].get("completion", ""),
            "coherence": coherence_result["answer"],
            "coherence_reasoning": coherence_result["reasoning"],
            "fluency": fluency_result["answer"],
            "fluency_reasoning": fluency_result["reasoning"]
        })
    
    print(f"--- Completed Evaluation. Evaluated {len(output)} samples. ---")
    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-name", type=str,
                        help="Name of the model to use.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input file.")
    parser.add_argument("--output-json-file", type=str,
                        help="Path to the output JSON file.")
    parser.add_argument("--max-concurrent-requests", type=int, default=10,
                        help="Maximum number of concurrent API requests.")
    parser.add_argument("--type", type=str, choices=["hyperparam_search", "main_generations"], required=True,
                        help="Type of evaluation to perform.")
    
    parsed_args = parser.parse_args()
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    semaphore = asyncio.Semaphore(parsed_args.max_concurrent_requests)
    
    if parsed_args.type == "hyperparam_search":
        asyncio.run(eval_hyperparam_search(parsed_args, client, semaphore))
    elif parsed_args.type == "main_generations":
        asyncio.run(eval_main_generations(parsed_args, client, semaphore))
    