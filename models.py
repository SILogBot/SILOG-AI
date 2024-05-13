import numpy
import tensorflow as tf
import tflearn
import json
from konlpy.tag import Okt
import random
# 형태소 분석기 초기화
okt = Okt()

# intents.json 파일을 불러옴
with open('intents.json',encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# 인텐트와 패턴을 처리
for intent in data['intents']:
    for pattern in intent['patterns']:
        # 한국어 토큰화
        tokens = okt.morphs(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# 중복을 제거하고 정렬
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    for w in words:
        bag.append(1 if w in doc else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# 배열로 변환
training = numpy.array(training)
output = numpy.array(output)

# 신경망 모델 설정
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(net)
# model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)
# model.save("model.tflearn")

# 단어 가방 생성 함수
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = okt.morphs(s)
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


# 은 음악기호입니다
def chat(inp):
    print("챗봇과 대화를 시작합니다.")
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

