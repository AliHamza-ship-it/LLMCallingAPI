import OpenAI from 'openai';
import dotenv from 'dotenv';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// 1. Load environment variables
dotenv.config();

const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey) {
    console.error('❌ ERROR: OPENROUTER_API_KEY not found in .env file.');
    console.error('📌 Please create a .env file with your key.');
    process.exit(1);
}

// 2. Initialize the OpenRouter client
const client = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: apiKey,
});

// 3. Simple cost estimator (Node doesn't have tiktoken easily, so we approximate)
function estimateCost(prompt) {
    // Rough estimate: 1 token ≈ 4 characters
    const inputTokens = Math.ceil(prompt.length / 4);
    const pricePerInputMillion = 0.15;
    const pricePerOutputMillion = 0.60;
    const estimatedOutputTokens = 300;

    const cost = (inputTokens * pricePerInputMillion / 1_000_000) +
        (estimatedOutputTokens * pricePerOutputMillion / 1_000_000);

    console.log(`📊 Estimated Input Tokens: ~${inputTokens}`);
    console.log(`💰 Estimated Max Cost: $${cost.toFixed(6)} (for ~300 output tokens)`);
    return inputTokens;
}

// 4. Define the conversation
const messages = [
    { role: 'system', content: 'You are a helpful AI assistant that explains code simply to a beginner.' },
    { role: 'user', content: 'Explain what an API is to a 10-year-old.' }
];

// 5. Estimate cost
estimateCost(messages[1].content);

console.log('\n🤖 Streaming Assistant Response: ');

// 6. Make the streaming call
try {
    const stream = await client.chat.completions.create({
        model: 'openai/gpt-oss-20b:free',
        messages: messages,
        temperature: 0.0,
        max_tokens: 300,
        stream: true,
    });

    let fullResponse = '';
    for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
            process.stdout.write(content);
            fullResponse += content;
        }
    }

    console.log('\n\n✅ Streaming complete.');
    console.log(`📝 Full response length: ${fullResponse.length} characters.`);

} catch (error) {
    console.error(`\n❌ API Call Failed: ${error.message}`);
    console.error('💡 Check your internet, API key, or OpenRouter credits.');
}