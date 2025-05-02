from openai import OpenAI
client = OpenAI(api_key='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')

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
