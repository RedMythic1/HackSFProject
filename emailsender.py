import os
import requests

def send_simple_message(to_email, body):
    # Make sure to set your Mailgun API key in your environment variables
    api_key = '14c5659395f6de328f9c70c96dc393b0-67bd41c2-12a40c25'  # Replace with your actual Mailgun API key

    if not api_key:
        raise ValueError("API_KEY environment variable not set.")

    response = requests.post(
        "https://api.mailgun.net/v3/sandboxa5cc6a7242d748f29c8ff458b5f780f5.mailgun.org/messages",
        auth=("api", api_key),
        data={
            "from": "Mailgun Sandbox <postmaster@sandboxa5cc6a7242d748f29c8ff458b5f780f5.mailgun.org>",
            "to": to_email,  # Use the provided email address
            "subject": "Hello from Mailgun",
            "text": body  # Use the provided body
        }
    )

    return response
