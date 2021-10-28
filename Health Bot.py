#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random


# In[13]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


# In[15]:


intents_file = open('/home/newton/Documents/Ai/Chatbot/intents.json').read()
intents = json.loads(intents_file)


# In[18]:


#preprocessing the data
words = []
classes = []
documents = []
ignore_letters = ['!','?',',','.']


# In[21]:


nltk.download('punkt')
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        #add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[22]:


print(documents)


# In[26]:


#lemmatizing to reduce the canonical words
nltk.download('wordnet')
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(len(documents), 'documents')
print(len(classes), 'classes', classes)
print(len(words), 'unique lemmatized words', words)

pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))


# In[31]:


#training the data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype = object)
train_x = list(training[:,0])
train_y = list(training[:,1])
print('Training data is created')


# In[33]:


model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation ='softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics =['accuracy'])

hist =model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.h5', hist)

print('model is created')


# In[42]:


#interacting with the chatbot
from keras.models import load_model
model = load_model('chatbot_model.h5')
intents = json.loads(open('/home/newton/Documents/Ai/Chatbot/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details = True):
    sentence_words = clean_up_sentence(sentence)
    bag =[0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                bag [i] = 1
                if show_details:
                    print('found in bag: %s' % word)
    return(np.array(bag))

def predict_class(sentence):
    p =bag_of_words(sentence, words, show_details = False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

#Creating tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get('1.0','end-1c').strip()
    EntryBox.delete('0.0', END)
    
    if msg != '':
        ChatBox.config(state = NORMAL)
        ChatBox.insert(END, 'You: ' + msg+ '\n\n')
        ChatBox.config(foreground = '#446665', font = ('Verdana', 12))
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
        ChatBox.insert(END, 'Newton: ' +res+ '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
        
root = Tk()
root.title('Talk2me')
root.geometry('400x500')
root.resizable(width=FALSE, height=FALSE)

ChatBox = Text(root, bd=0, bg = 'white', height = '8', width = '50', font = 'Arial',)
ChatBox.config(state=DISABLED)
scrollbar = Scrollbar(root, command = ChatBox.yview, cursor = 'heart')
ChatBox['yscrollcommand']=scrollbar.set
SendButton = Button(root, font=('Verdana', 12, 'bold'), text ='Send', width = '12', height = 5, bd = 0, bg = '#f9a602', activebackground = '#3c9d9b', fg = '#000000', command = send)
EntryBox = Text(root, bd=0, bg='white', width='29',height='5', font = 'Arial')
scrollbar.place(x=376,y=6,height=386)
ChatBox.place(x=6,y=6,height=386,width=370)
EntryBox.place(x=128,y=401, height=90, width=265)
SendButton.place(x=6,y=401,height=90)

root.mainloop()


# In[ ]:




