import os
import sys
import json
import boto3
import pathlib
import requests 

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

# Initialize Bedrock runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"

# Build the request payload
body = json.dumps({
    "prompt": "Summarize the main themes of Shakespeare's Hamlet in two sentences.",
    "max_tokens_to_sample": 500
})

# Invoke the model
response = client.invoke_model(
    modelId=model_id,
    body=body
)

# Parse and print the output text
model_response = json.loads(response["body"].read())
output_text = model_response["results"][0]["outputText"]
print(output_text)
