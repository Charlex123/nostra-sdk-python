# NostraHealthAI Python SDK

Official Python SDK for **NostraHealthAI-1.0** - Advanced Medical AI Assistant.

## Installation

```bash
pip install nostrahealthai-sdk
```

## Quick Start

```python
from nostra_healthai import NostraHealthAI

# Initialize client
client = NostraHealthAI(api_key='your-firebase-token')

# Chat with the AI
response = client.chat(message='What are the symptoms of high blood pressure?')
print(response['response'])
print(f"Model: {response['modelInfo']['name']}")

# Analyze medical file
job_id = client.analyze_file('./lab-results.png')
result = client.wait_for_job_completion(job_id)

print(f"Risk Level: {result['data']['riskLevel']}")
print(f"Summary: {result['data']['summary']}")
```

## Features

- 🩺 **Medical Chat** - Conversational AI with context memory
- 📊 **File Analysis** - Analyze lab reports, X-rays, medical images
- 🤖 **Multi-Model AI** - Powered by OpenAI GPT-4o + Google Gemini
- 📚 **Evidence-Based** - Cites medical sources (RxNorm, OpenFDA, PubMed)
- ⚡ **Async Support** - Built-in job polling and completion waiting

## Documentation

Full documentation available at: [SDK Documentation](../README.md)

## Examples

See [python-example.py](../examples/python-example.py) for comprehensive examples.

## Requirements

- Python 3.8+
- requests >= 2.31.0

## License

MIT License
