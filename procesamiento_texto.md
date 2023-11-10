## Tokenization
import nltk
sentences = nltk.sent_tokenize("TEXT")
words = nltk.word_tokenize("TEXT")
## Stemming (stopwords antes)
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
words = [PorterStemmer.stem(x) for x in words]
sentences[i] = ' '.join(words)
stemmer.stem("cried")
## Lemmatization 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(x) for x in words]
sentences[i] = ' '.join(words)
## Removal of the stop words
from nltk.corpus import stopwords
words = [x for x in words if x not in stopwords.words('english')]
## POS tagging (antes de lematizacion)
my_words_tagged = nltk.pos_tag(my_words)
## Visualize the text data as a WordCloud
from wordcloud import WordCloud
from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wc = WordCloud(background_color='white', max_words=30)              # Customize the output.
wc.generate(a_long_sentence)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")                                    # Turn off the axes.
plt.show()
## Padding
from tensorflow.keras.preprocesing.sequence import pad_sequences
// Hay que rellenar con ceros para optimizar, (se crea una matriz)
## One-Hot Encoding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index)

#####
- Se puede codificar las palabras por numero de apariciones
- Se cuentan las frecuencias de cada palabra, se ordenan por frecuencias/mayor probabilidad y se numeran
- Nos quedamos con las palabras mas comunes
sorted(Counter(np.concatenate(np.array(sentences,dtype=object))).items(), key=lambda x:x[1], reverse=True)
words = sum(sentences, [])
vocab = Counter(words)
vocab.most_common(vocab_size)
from nltk import FreqDist
vocab = FreqDist(np.hstack(sentences))
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) 



