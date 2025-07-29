import asyncio
import edge_tts


# Function to generate speech
async def speak(text, voice, filename):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(filename)
    print(f"Saved: {filename}")


# Main function to run both voices
async def main():
    # Voice: Neerja (Female Indian English)
    await speak("Hello, Welcome to Titan customer support care, How can i help you today?", "en-IN-NeerjaNeural", "neerja.mp3")

    # Voice: Prabhat (Male Indian English)
    await speak("Hello, Welcome to Titan customer support care, How can i help you today?", "en-IN-PrabhatNeural", "prabhat.mp3")


# Run the async main
asyncio.run(main())
