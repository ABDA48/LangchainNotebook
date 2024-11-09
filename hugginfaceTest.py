from transformers import pipeline
import os
from dotenv import load_dotenv,find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from transformers import pipeline
load_dotenv(find_dotenv())
from gtts import gTTS
import os
import requests
HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")

def imageTotext(url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text=pipe(url)
    print(text)
    return text[0]['generated_text']



def generate_scenario_openAI(scenario):
    template=""" 
     You are a story teller> You can generate a short story basedon a simple narrative,the story should be more than 20 words ;
     CONTEXT:{scenario}
     STORY:
    """
    prompt=PromptTemplate(input_variables=["scenario"], template=template)
    story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    story=story_llm.run(scenario=scenario)
    print(story)
    return story


#generate_scenario_openAI(text)

generator = pipeline("text-generation", model="gpt2",device=0)
def generate_scenario(scenario):
    story = generator(f"take this {scenario} as topic and generate one paragraph of story.make the story narrative .")
    print(story[0]['generated_text'])
    return story[0]['generated_text']

#generate_scenario(text)
""" 
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
output = query({
	"inputs": "Can you please let us know more details about your ",
})
print(output)


from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
pipe(messages)
"""

from huggingface_hub import InferenceClient

client = InferenceClient(api_key=HUGGINGFACE_TOKEN)
topic=imageTotext('bts.jpg')
messages = [
	{
		"role": "user",
		"content": f"can u create a story about this topic {topic} and make it in 1 paragraph only"
	}
]

stream = client.chat.completions.create(
    model="mistralai/Mistral-Nemo-Instruct-2407", 
	messages=messages, 
	max_tokens=500,
	stream=True
)
text=''
for chunk in stream:
    text += chunk.choices[0].delta.content

assert text, "No text to speak"
tts = gTTS(text=text, lang='en')

# Save the audio file
tts.save("output.mp3")

# Play the audio (optional, depends on your OS)
os.system("start output.mp3")