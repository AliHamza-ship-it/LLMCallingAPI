import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from datetime import datetime

# =============================================
# 1. Load environment and setup client
# =============================================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# =============================================
# 2. Define the 10 Bad Prompts (Day 1 style)
# =============================================
BAD_PROMPTS = [
    "Write about climate change.",
    "Tell me how to bake a cake.",
    "Explain AI.",
    "Write code for a website.",
    "What is the meaning of life?",
    "Give me a workout plan.",
    "Translate this: Hello, how are you?",
    "Write a story.",
    "Summarize this article: https://www.newyorker.com/news/news-desk/andrew-cuomos-war-against-a-federal-prosecutor",
    "Tell me about space."
]

# =============================================
# 3. Define the 10 Fixed Prompts (Day 2 CTCO + Persona)
# =============================================
FIXED_PROMPTS = [
    "[Context: I am a high school student preparing for a debate against a climate skeptic.] [Task: Write a 500-word opening argument.] [Constraints: Use exactly 3 scientific statistics. Tone must be confident and factual, not emotional. Avoid complex chemistry jargon.] [Output Format: 5 paragraphs – intro, 3 body paragraphs (each with one stat), conclusion.]",

    "[Context: I am a 10-year-old baking for the first time, with my mom's supervision.] [Task: Provide step-by-step instructions for a simple vanilla sponge cake.] [Constraints: No nuts, no electric mixers (hand whisk only), total time under 1 hour.] [Output Format: Numbered steps, with a separate 'prep list' at the top.]",

    "[Context: My grandmother is 80 and uses a flip phone. She thinks AI is a robot apocalypse.] [Task: Explain what a Large Language Model actually does.] [Constraints: Use only analogies (e.g., 'autocomplete on steroids'), absolutely zero technical terms like 'neural networks' or 'transformers'. Max 150 words.] [Output Format: A single, warm paragraph written like a letter to her.]",

    "[Context: I am building a portfolio site for my photography.] [Task: Write HTML, CSS, and vanilla JavaScript for a responsive navigation bar that sticks to the top.] [Constraints: Mobile-first design. Dark theme. Must include a hamburger menu for screens under 768px.] [Output Format: Provide all three languages in separate code blocks with comments explaining each section.]",

    "[Context: I am writing a comparative religion essay for college.] [Task: Compare the existentialist view (Sartre) vs. the nihilist view (Nietzsche) on the meaning of life.] [Constraints: Objective tone. 300 words exactly. Cite one primary text source for each.] [Output Format: A table with two columns: 'Existentialism' and 'Nihilism', with rows for 'Definition', 'Purpose', and 'Actionable advice'.]",

    "[Context: I am a 45-year-old office worker who hasn't exercised in 5 years.] [Task: Design a 4-week beginner bodyweight workout routine.] [Constraints: Only 30 minutes per day. No jumping (bad knees). Focus on mobility and core strength.] [Output Format: A weekly calendar (Mon-Sun) with exercises named and reps/sets listed.]",

    "[Context: This is a legally binding rental agreement for a landlord in Quebec.] [Task: Translate this full document from English to Canadian French.] [Constraints: Must maintain legal precision. Do not translate proper nouns or dates. Keep the exact paragraph numbering.] [Output Format: Plain text with the exact same structure as the original, with a caveat note at the top that says 'This is a translation, refer to original for legal validity'.]",

    "[Context: I need a bedtime story for my 4-year-old daughter who loves unicorns and is scared of thunder.] [Task: Write a 500-word story about a unicorn who learns to make friends with the thunderclouds.] [Constraints: The moral must be 'facing your fears'. Use short sentences. Include a happy ending. No scary villains.] [Output Format: Three short chapters with titles. Use dialogue between the unicorn and a raindrop.]",

    "[Context: I am a busy product manager deciding if this AI research paper is relevant to our new feature.] [Task: Summarize the key findings.] [Constraints: Focus strictly on the 'Results' and 'Limitations' sections. Ignore the introduction. Max 3 bullet points.] [Output Format: Bullet points. First bullet: 'What it does'. Second: 'How well it works'. Third: 'Why we can't use it yet'.]",

    "[Context: I am an 8th-grade student doing a science project on astronomical objects.] [Task: Explain the life cycle of a star, from nebula to black hole.] [Constraints: Avoid complex math. Use analogies to everyday objects (e.g., 'a star is like a pressure cooker'). Max 400 words.] [Output Format: 4 sections with headings: 'Birth', 'Life', 'Death', and 'Afterlife'. Include a simple ASCII diagram showing the stages.]"
]

# =============================================
# 4. Helper: Estimate tokens + cost
# =============================================
def estimate_cost(prompt, model="openai/gpt-4o-mini"):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(encoding.encode(prompt))
    except Exception:
        input_tokens = len(prompt) // 4

    price_per_input_million = 0.15
    price_per_output_million = 0.60
    estimated_output_tokens = 500  # Increased because outputs are longer now
    cost = (input_tokens * price_per_input_million / 1_000_000) + \
           (estimated_output_tokens * price_per_output_million / 1_000_000)
    
    return input_tokens, cost

# =============================================
# 5. Core function: Send prompt to AI (non-streaming for easy saving)
# =============================================
def get_response(user_prompt, system_message="You are a helpful AI assistant."):
    """Send a prompt and return the full response text."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",  # Super reliable. Change to "meta-llama/llama-3.1-8b-instruct:free" if you want $0 cost.
            messages=messages,
            temperature=0.0,       # 0.0 = deterministic, best for comparing bad vs good fairly
            max_tokens=1000,       # Increased from 100 so it can write essays
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API ERROR: {str(e)}"

# =============================================
# 6. Main Execution: Run the comparison
# =============================================
def run_comparison():
    print("🚀 Day 2: Bad vs Fixed Prompts Comparison Engine")
    print("=" * 60)
    
    # Open a text file to save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_comparison_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Prompt COMPARISON: BAD PROMPTS vs FIXED (CTCO) PROMPTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Loop through each pair (Bad vs Fixed)
        for i in range(10):
            bad_prompt = BAD_PROMPTS[i]
            good_prompt = FIXED_PROMPTS[i]
            
            print(f"\n📌 Processing Pair {i+1}/10...")
            f.write(f"\n{'='*80}\n")
            f.write(f"PAIR {i+1}\n")
            f.write(f"{'='*80}\n\n")
            
            # ---- Bad Prompt ----
            print(f"  ⚠️  Running BAD prompt {i+1}...")
            input_tokens_bad, cost_bad = estimate_cost(bad_prompt)
            f.write(f"--- BAD PROMPT {i+1} ---\n")
            f.write(f"PROMPT: {bad_prompt}\n")
            f.write(f"INPUT TOKENS: {input_tokens_bad} | EST. COST: ${cost_bad:.6f}\n")
            f.write("-" * 60 + "\n")
            
            bad_output = get_response(bad_prompt)
            f.write(f"OUTPUT:\n{bad_output}\n\n")
            
            # ---- Good Prompt ----
            print(f"  ✅ Running FIXED prompt {i+1}...")
            input_tokens_good, cost_good = estimate_cost(good_prompt)
            f.write(f"--- FIXED PROMPT {i+1} ---\n")
            f.write(f"PROMPT: {good_prompt}\n")
            f.write(f"INPUT TOKENS: {input_tokens_good} | EST. COST: ${cost_good:.6f}\n")
            f.write("-" * 60 + "\n")
            
            good_output = get_response(good_prompt)
            f.write(f"OUTPUT:\n{good_output}\n\n")
            
            # ---- Brief analysis ----
            token_diff = input_tokens_good - input_tokens_bad
            f.write(f"📊 TOKEN DIFFERENCE: Fixed uses {token_diff} more input tokens (this is the cost of adding CTCO).\n")
            f.write("-" * 80 + "\n")
            
            print(f"  ✅ Saved to {filename}")
    
    print("\n" + "=" * 60)
    print(f"🎉 ALL DONE! Results saved to: {filename}")
    print(f"📂 Open this file and scroll through to see the MASSIVE difference.")
    print("💡 Notice how 'Bad' outputs are generic and rambling, while 'Fixed' outputs are precise, structured, and tailored.")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()