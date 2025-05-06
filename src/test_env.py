# from dotenv import load_dotenv
# import os

# load_dotenv()

# print("API Key Loaded:", os.getenv("OPENAI_API_KEY"))
# from dotenv import load_dotenv
# import os
# from pathlib import Path

# env_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=env_path)

# print("API Key Loaded:", os.getenv("OPENAI_API_KEY"))


import os
os.environ["OPENAI_API_KEY"] = "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"
print("Manually set key:", os.getenv("OPENAI_API_KEY"))
