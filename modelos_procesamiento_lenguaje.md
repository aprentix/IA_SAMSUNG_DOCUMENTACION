## PROCESAMIENTO DE TEXTO
.lower()
.upper()
.split(" ")
" ".join(list)
[ .?{m,n}^$| ]
re.compile(expresion)
re.sub(expresion, sustituto)
re.search(expresion, frase) # puedes usar group()
## GET INFO FROM A WEB-PAGE
import requests as rq
import bs4
rq.get(url) => .headers.keys()
soup = bs4.BeautifulSoup(res.text, "html.parser").prettify()
## PREPROCESAMIENTO DE TEXTO
import nltk
### Stopwords
from nltk.corpus import stopwords
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [x for x in words if x not in stopwords.words('english')]
    sentences[i] = ' '.join(words)
### Stemming
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
sentences = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(tweet)
words = [stemmer.stem(x) for x in words] # hacer stemming de las palabras tokenizadas
### Lematizar
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])              # Tokenize into words.
    words = [lemmatizer.lemmatize(x) for x in words]      # Lemmatization. 
    sentences[i] = ' '.join(words)
### POS Tagging
my_words_tagged = nltk.pos_tag(my_words)
### WORDCLOUD
from nltk.stem import WordNetLemmatizer
from PIL import Image
wordcloud.WordCloud(background_color='white', max_words=30)              # Customize the output.
wc.generate(a_long_sentence)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")                                    # Turn off the axes.
plt.show()
## MODELOS DE PROCESAMIENTO DE TEXTO
sklearn.feature_extraction.text.TfidfVectorizer(max_features = 1000, min_df = 2, max_df = 0.6, stop_words = stopwords.words('english'))
sklearn.feature_extraction.text.CountVectorizer(max_features = 15, min_df = 1, max_df = 3, stop_words = list(ENGLISH_STOP_WORDS))
X = vectorizer.fit_transform(my_docs).toarray()
## Redes neuronales
my_model = tf.keras.Sequential() ## Vamos a ir construyendo la red neuronal
my_model.add(tf.keras.layers.Embedding(n_words, n_input)) # n_words = vocabulary size, n_input = dimension of the embedding space.
my_model.add(tf.keras.layers.LSTM(units=n_neurons, return_sequences=False, input_shape=(None, n_input), activation='tanh'))
my_model.add(Dense(n_cat, activation='softmax'))
my_model.summary()
my_optimizer=Adam(learning_rate=learn_rate)
my_model.compile(loss = "categorical_crossentropy", optimizer = my_optimizer, metrics=["accuracy"])
