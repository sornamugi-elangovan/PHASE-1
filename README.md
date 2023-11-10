IBM-nm project 1
CREATING CHATBOT USING PYTHON
 
TABLE OF CONTENT :

# problem statement 
# design thinking process
# phases of development.
# libraries 
# NLP technique
# chatbot interaction with users 
# the web application.
# innovative techniques or approaches

PROBLEM STATEMENT :
               
 Creating a chatbot using Python involves designing an interactive conversational agent capable of understanding user input and providing appropriate responses. The chatbot should be able to handle natural language processing, including tasks such as text preprocessing, intent recognition, and generating contextually relevant and meaningful replies. Additionally, the chatbot should be able to integrate with various platforms and APIs, enabling seamless communication with users.

DESIGN THINKING PROCESS :

Design thinking process for creating a chatbot using Python typically involves the following steps:

Empathize: Understand the target audience, their needs, and preferences for an optimal user experience.

Define: Clearly define the purpose and goals of the chatbot, outlining the specific problems it aims to solve and the value it will provide to users.

Ideate: Brainstorm potential features, conversation flows, and user interactions that align with the defined purpose and goals.

Prototype: Create a basic version of the chatbot to test and refine the core functionalities and interactions, using Python libraries such as NLTK or spaCy for natural language processing.

Test: Collect feedback from users and stakeholders, and iterate on the chatbot design to improve its effectiveness and user satisfaction.

Implement: Develop the final version of the chatbot, integrating it with appropriate APIs, databases, and other systems as needed, using Python frameworks like Flask or Django.

Iterate: Continuously monitor and enhance the chatbot's performance, incorporating user feedback and data-driven insights to refine its functionality and usability over time.  

DEVELOPMENT PHASE :
      Loading and preprocessing the dataset

you can follow these steps:

 Load the dataset. You can use any Python library to load your dataset, such as pandas or NumPy.

Dataset Link: https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot
  

 Preprocess the dataset. This involves cleaning the data, removing 
stop words, and converting the data to a format that can be used 
by your chatbot model.
 
Import Necessary Libraries: 
Start by importing the required 
libraries:
import random
import json
import nltk
from nltk.stem 
import WordNetLemmatizer
from tensorflow.keras.models 
import load_model
import numpy as np

Load and Preprocess the Dataset: 
        
For this example, let's use a 
JSON file containing predefined intents and responses. Each intent 
contains a list of patterns and responses. You can create a JSON file like this

{
"intents": [
 {
 "tag": "greeting",
 "patterns": ["Hello", "Hi", "Hey"],
 "responses": ["Hello!", "Hi there!", "Hey!"]
 },
 {
 "tag": "goodbye",
 "patterns": ["Goodbye", "Bye", "See you later"],
 "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
 },
 {
 "tag": "name",
 "patterns": ["What's your name?", "Who are you?"],
 "responses": ["I'm a chatbot.", "I'm ChatGPT, a chatbot."]
 }
 // Add more intents as needed
 ]
}

You can load this JSON file in Python as follows:
with open('intents.json') as file:
 intents = json.load(file)

Preprocess the Dataset: Preprocess the dataset to prepare it for training. This typically involves tokenization, lemmatization, and creating training data.

# Extract patterns and responses

patterns = []
responses = []
for intent in intents['intents']:
for pattern in intent['patterns']:
patterns.append(pattern)
responses.append(intent['tag'])

# Tokenization and Lemmatization

lemmatizer = WordNetLemmatizer()
words=nltk.word_tokenize("".join(patterns))
words=[lemmatizer.lemmatize(word.lower() for word in words]

# Create training data

training_data = []
output = [0] * len(intents['intents'])
for i, intent in enumerate(intents['intents']):
output[i] = 1
for pattern in intent['patterns']:
bag = [0] * len(words)
pattern_words=nltk.word_tokenize(pattern)
 pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
for p in pattern_words:
for j, w in enumerate(words):
if w == p:
bag[j] = 1
training_data.append([bag, output.copy()])
output[i] = 0
from tensorflow.keras.models 
import Sequential
from tensorflow.keras.layers 
importDensemodel=Sequential()
model.add(Dense(128,input_shape=(len(training_data[0][0]),),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(ln(training_data[0][1]), activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
X = np.array([i[0] for i in training_data])
y = np.array([i[1] for i in training_data])
model.fit(X, y, epochs=100, batch size=5)
model.save('chatbot_model.h5')
 
Create a Chat Function: Create a function to interact with the chatbot.

def chat():
print("Start chatting with the bot(type'quit'to exit):")
 while True:
 user_input = input("You: ")
 if user_input.lower() == 'quit':
 break
user_input=nltk.word_tokenize(user_input) user_input=[lemmatizer.lemmatize(word.lower()) for word in user_input]
 bag = [0] * len(words)
 for word in user_input:
 for i, w in enumerate(words):
 if w == word:
 bag[i] = 1
prediction=model.predict(np.array([bag]))[]
predicted_intent=intents['intents'][np.argmax(prediction)]['tag']
for intent in intents['intents']:
 if intent['tag'] == predicted_intent:
 responses = intent['responses']
 print("Bot:", random.choice(responses))
 
Run the Chatbot: Call the chat function to run the chatbot.
chat()

PROGRAM
[1]: import numpy as np 
# linear algebra
import pandas as pd 
# data processing, 
CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory 
# For example,running this (by clicking run or pressing Shift+Enter) will list␣ ↪all files under the input directory
import os
for dirname, _, filenames in os.walk('dialogs.txt'):
for filename in filenames:
print(os.path.join(dirname, filename))
[2]: import numpy as np
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
[3]: f = open("dialogs.txt","r",errors = 'ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
[4]: nltk.download('punkt')
nltk.download('wordnet')
[nltk_data] Downloading package punkt to
[nltk_data] C:\Users\rohig\AppData\Roaming\nltk_data…
[nltk_data] Package punkt is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data] C:\Users\rohig\AppData\Roaming\nltk_data…
[nltk_data] Package wordnet is already up-to-date!
[4]: True
1
[5]: sent_token = nltk.sent_tokenize(raw_doc)
word_token = nltk.word_tokenize(raw_doc)
print(f"Number of sentence : {len(sent_token)}")
print(f"Number of Words in : {len(word_token)}")
Number of sentence : 8272
Number of Words in : 60702
[6]: lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = {ord(punct):None for punct in string.punctuation} def
lem_token(tokens):
return [lemmer.lemmatize(token) for token in tokens]
def lem_normalize(text):
return lem_token(nltk.word_tokenize(text.lower().
↪translate(remove_punct_dict)))
[7]: GREET_INPUT = ("hello,hi,sup")
GREET_RESPONSE = ['Hi',"Hello","I am glad you are talking to me!"]
def greet(sentence):
for word in sentence.split():
if word.lower() in GREET_INPUT:
return random.choice(GREET_RESPONSE)
[8]: def response(user_response):
sent_token.append(user_response)
tfidfvec = TfidfVectorizer(tokenizer = lem_normalize,stop_words = 'english') tfidf =
tfidfvec.fit_transform(sent_token)
vals = cosine_similarity(tfidf[-1],tfidf)
idx = vals.argsort()[0][-2]
flat = vals.flatten()
flat.sort()
req_tfidf = flat[-2]
sent_token.remove(user_response)
if (req_tfidf == 0):
return " I am Sorry! I dont understand you"
else:
return str(sent_token[idx])
[9]: flag = True
print("BOT : My Name is BOThi, Let's Have Conversation! If you want to exit any␣
↪time, just type Bye! ")
print("\n")
while (flag==True):
user_response = input("You : ")
2
if (user_response != "bye"):
if (user_response == 'thanks'):
flag = False
print("BOT : You are welcome..")
else:
if (greet(user_response) != None):
print("BOT :" +"\t"+greet(user_response))
else:
print("BOT L",end = "")
print(response(user_response))
print("\n")
else:
flag = False
print("BOT : Goodbye! Take Care")
BOT : My Name is BOThi, Let's Have Conversation! If you want to exit any time, justtype Bye!
You : hello
BOT : I am glad you are talking to me!
You : what is python
D:\Anacinda\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning:
The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
warnings.warn(
D:\Anaconda\Lib\site-packages\sklearn\feature_extraction\text.py:408:
UserWarning: Your stop_words may be inconsistent with your preprocessing.
Tokenizing the stop words generated tokens ['ha', 'le', 'u', 'wa'] not in
stop_words.
warnings.warn(
I am Sorry! I dont understand you
You : what happend to you
BOT L I am Sorry! I dont understand you
You : what
BOT L I am Sorry! I dont understand you
3
You : why
BOT L I am Sorry! I dont understand you
You : have a lunch
BOT Land then i made lunch.
You : are you feel good
BOT Lgood!
You : then thank you
BOT Lthank you very much.
You : hi
BOT : Hello
You : bye
BOT : Goodbye! Take Care

Conclusion :
Developing a chatbot using Flask entails creating a robust and
interactive conversational interface that can understand user inputs and
provide relevant and coherent responses. By implementing key features
such as input processing, intent recognition, context management,
response generation, and user experience enhancement, you can
ensure a seamless and engaging chatbot experience.
Through model training, evaluation, and continuous refinement, you can
optimize the chatbot's performance and enhance its ability to handle a
variety of user queries and interactions. Integrating the chatbot into a
Flask web application enables real-time communication between the
user interface and the chatbot's backend, facilitating an efficient and
user-friendly experience.
Overall, the combination of Flask and a well-trained chatbot model can
provide a powerful and versatile platform for building intelligent and
effective conversational agents that can cater to a diverse range of user needs and performances
 
 




