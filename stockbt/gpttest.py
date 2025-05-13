import g4f

# Enable debug logging (optional)
g4f.debug.logging = False

# Disable automatic version checking (optional)
g4f.debug.version_check = False

# Normal (non-streaming) response
response = g4f.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello, how are you today?"}],
    provider=g4f.Provider.PollinationsAI,
)
print(response)