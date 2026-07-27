import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file.")
    print("📌 Please create a .env file with your key.")
    sys.exit(1)

# 2. Initialize the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# ============================================
# 🆕 INSERT THE MODEL LISTING CODE RIGHT HERE
# ============================================
# print("🔍 Fetching available free models from OpenRouter...")
# try:
#     free_models = client.models.list()
#     print("✅ Free models available:")
#     for model in free_models:
#         # The ID is what you put in the "model=" parameter
#         if ":free" in model.id:  
#             print(f"  - {model.id}")
# except Exception as e:
#     print(f"⚠️ Could not fetch models: {e}")

def estimate_cost(prompt, model="openai/gpt-4o-mini"):
    """Estimate input token count and cost before calling the API."""
    try:
        # tiktoken uses 'cl100k_base' for most GPT-4/3.5 models
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(encoding.encode(prompt))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        input_tokens = len(prompt) // 4

    # OpenRouter pricing for gpt-4o-mini (as of 2025)
    price_per_input_million = 0.15   # $0.15 per 1M input tokens
    price_per_output_million = 0.60  # $0.60 per 1M output tokens
    
    # We assume ~300 output tokens for estimation (safe average)
    estimated_output_tokens = 300
    cost = (input_tokens * price_per_input_million / 1_000_000) + \
           (estimated_output_tokens * price_per_output_million / 1_000_000)
    
    print(f"📊 Estimated Input Tokens: {input_tokens}")
    print(f"💰 Estimated Max Cost: ${cost:.6f} (for ~300 output tokens)")
    return input_tokens

# 3. Define the conversation (System + User)
messages = [
    {"role": "system", "content": "You are a helpful AI assistant that explains code simply to a beginner."},
    {"role": "user", "content": "Explain what an API is to a 10-year-old."}
]

# 4. Estimate cost before hitting the API
estimate_cost(messages[1]["content"])

print("\n🤖 Streaming Assistant Response: ", end="", flush=True)

try:
    # 5. Make the streaming call
    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",  # Cheapest, fastest, reliable
        messages=messages,
        temperature=0.0,              # Balanced creativity
        max_tokens=100,               # Limits the response length
        stream=True,                  # <-- Production standard
    )

    full_response = ""
    for chunk in stream:
        # Safely extract content from the chunk
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    if not full_response:
        print("\n⚠️ The AI generated an empty response. Try increasing `max_tokens`.")
    print("\n\n✅ Streaming complete.")
    print(f"📝 Full response length: {len(full_response)} characters.")

except Exception as e:
    print(f"\n❌ API Call Failed: {e}")
    print("💡 Check your internet, API key, or OpenRouter credits.")