import tkinter as tk
from tkinter import ttk

class coding(tk.Tk):
  def __init__(self):
    super().__init__()
    self.title('의미 원소 기반 AI')
    self.geometry('400x300')
    self.resizable(width=False, height=False)
    self._init_ui()
  def _init_ui(self):
    ttk.Label(self, text = '의미 원소 기반 AI', font = ('굴림', 20, 'bold')).pack()
  
    #엔트리 추가
    self.entry = ttk.Entry(self, font = ('굴림', 10))
    self.entry.pack()
    ttk.Button(self, command = self.entry_saying).pack()
  
  def entry_saying(self):
    self.entry1 = self.entry.get()
    print(self.entry1)

    

if __name__ == '__main__':
  app = coding()
  app.mainloop()

print(app.entry1)
A = app.entry1




#실체
I = 1
YOU = 2
SOMEONE = 3
SOMETHING = 4
THING = 5
PEOPLE = 6
BODY = 7

#관계&지시
THIS = 8
THESAME = 9
OTHER = 10

#수량
ONE = 11
TWO = 12
SOME = 13
ALL = 14
MANY = 15
MUCH = 16

#평가
GOOD = 17
BAD = 18

#기술
BIG = 19
SMALL = 20

#정신 상태&인지
THINK = 21
KNOW = 22
WANT = 23
FEEL = 24
SEE = 25
HEAR = 26

#말하기
SAY = 27
WORDS = 28
TRUE = 29

#행동&사전
DO = 30
HAPPEN = 31
MOVE = 32

#존재&소유
SOMEWHERE = 33
THEREIS = 34
HAVE = 35

#생명&죽음
LIVE = 36
DIE = 37

#시간
WHEN = 381
NOW = 382
BEFORE = 39
AFTER = 40
ALONGTIME = 41
ASHORTTIME = 42
FORSOMETIME = 43
MOMENT = 44

#공간
WHERE = 45
HERE = 46
ABOVE = 47
BELOW = 48
FAR = 49
NEAR = 50
SIDE = 51
INSIDE = 52

#논리 연결
NOT = 53
MAYBE = 54
CAN = 55
BECAUSE = 56
IF = 57

#강화&유사
VERY = 58
LIKE = 59

# 리스트 정의 (문자열 없이 변수 그대로)
give = [DO, MOVE, SOMETHING, OTHER, SOMEONE]
eat = [DO, SOMETHING, SOMETHING, FEEL, GOOD]
want = [WANT, SOMETHING]
see = [SEE, SOMETHING]
hear = [HEAR, SOMETHING]
think = [THINK, SOMETHING]
know = [KNOW, SOMETHING]
feel = [FEEL, SOMETHING]
have = [HAVE, SOMETHING]
take = [DO, MOVE, SOMETHING]
make = [DO, SOMETHING]
say = [SAY, WORDS]
run = [MOVE, SOMEONE]
walk = [MOVE, SOMEONE]
sleep = [DO, SOMEONE]
love = [FEEL, SOMEONE]
hate = [FEEL, SOMETHING, BAD]
read = [SEE, WORDS]
write = [DO, WORDS]
look = [SEE, SOMETHING]
play = [DO, SOMETHING]
work = [DO, SOMETHING, MANY]
study = [DO, SOMETHING, KNOW]
listen = [HEAR, WORDS]
speak = [SAY, WORDS]
help = [DO, OTHER, SOMEONE, GOOD]
buy = [DO, MOVE, SOMETHING, OTHER, SOMEONE]
sell = [DO, MOVE, SOMETHING, OTHER, SOMEONE]
open = [DO, MOVE, SOMETHING]
close = [DO, MOVE, SOMETHING]
runfast = [MOVE, SOMEONE, VERY]
jump = [MOVE, SOMEONE]
sit = [DO, SOMEONE, HERE]
stand = [DO, SOMEONE, HERE]
eatapple = [DO, SOMETHING, SOMEONE, FEEL, GOOD]
drinkwater = [DO, SOMETHING, SOMEONE, FEEL, GOOD]
thinkidea = [THINK, SOMETHING, VERY]
knowfact = [KNOW, SOMETHING, TRUE]
wantmoney = [WANT, SOMETHING]
feelsad = [FEEL, SOMETHING, BAD]
seebird = [SEE, SOMETHING]
hearmusic = [HEAR, SOMETHING, GOOD]
saytruth = [SAY, WORDS, TRUE]
movebox = [MOVE, SOMETHING]
carry = [DO, MOVE, SOMETHING, OTHER, SOMEONE]
writeletter = [DO, WORDS, OTHER, SOMEONE]
readbook = [SEE, WORDS, SOMETHING]
lovefriend = [FEEL, SOMEONE, GOOD]
hateenemy = [FEEL, SOMEONE, BAD]
travel = [MOVE, SOMEONE, FAR]

# 주어
I = [I]
you = [YOU]
someone = [SOMEONE]
people = [PEOPLE]

# 동사
do = [DO, SOMETHING]
move = [MOVE, SOMETHING]
have = [HAVE, SOMETHING]
say = [SAY, WORDS]
see = [SEE, SOMETHING]
hear = [HEAR, SOMETHING]
think = [THINK, SOMETHING]
feel = [FEEL, SOMETHING]
want = [WANT, SOMETHING]

# 목적어
something = [SOMETHING]
other = [OTHER, SOMEONE]
good = [GOOD]
bad = [BAD]
words = [WORDS]

# 시간/장소 표현용 단어
now = [NOW]
here = [HERE]
there = [THEREIS]
before = [BEFORE]
after = [AFTER]

#여기서 부터

import random
from difflib import SequenceMatcher

SIMILARITY_THRESHOLD = 0.7  # 70% 이상이면 교체

# 단어 리스트 전부 가져오기
# globals()에서 리스트 변수만 선택
all_words = {k: v for k, v in globals().items() if isinstance(v, list)}

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def correct_and_convert(sentence):
    words = sentence.split()
    corrected = []

    for w in words:
        if w in all_words:
            corrected.append(w)
        else:
            # 오타 교정: 리스트 변수 이름과 유사율 계산
            sims = [(word, similar(w, word)) for word in all_words.keys()]
            max_sim = max(sims, key=lambda x: x[1])[1]
            candidates = [word for word, sim in sims if sim == max_sim]
            if max_sim >= SIMILARITY_THRESHOLD:
                corrected.append(random.choice(candidates))
            else:
                # 임계값 미만이면 제거
                pass

    # 사전에 있는 단어만 남기고 의미원소 숫자로 변환
    meaning_numbers = []
    for w in corrected:
        meaning_numbers.extend(all_words[w])
    return meaning_numbers

# 테스트 문장
sentence = A
# A 만들기
A = correct_and_convert(sentence)

print("A 숫자 결과")
print(A)


total_count = (len(A)+1)*len(A)/2
print(total_count)

tkwfsh_list = [THINK, KNOW, WANT, FEEL, SEE, HEAR]

for i in range(1, 7):
    globals()[f"positions{i}"] = [s for s, v in enumerate(A) if v == tkwfsh_list[i-1]]
    print(globals()[f"positions{i}"])

for i in range(1, 7):
    if  tkwfsh_list[i-1] in A:
      print(0)
      print(A.count(tkwfsh_list[i-1]))
      tkwfsh_list[i-1] = A.count(tkwfsh_list[i-1])

all_positions = positions1 + positions2 + positions3 + positions4 + positions5 + positions6


#정신 상태&인지 의미 원소 비율
for i in range(1, 7):
    if len(all_positions) == 0:
        print(0)
        globals()[f"ppa{i}"] = 0
    else:
        print(len(globals()[f"positions{i}"])/len(all_positions))
        globals()[f"ppa{i}"] = len(globals()[f"positions{i}"])/len(all_positions)


#positions1x = sum(positions1)/total_count
#위치 가중치
print('위치 가중치')
positions_listz = [0]*6
positions_list = [positions1, positions2, positions3, positions4, positions5, positions6]

for i in range(1, 7):
  positions_listz[i-1] = sum(positions_list[i-1])/total_count
  globals()[f"positions{i}x"] = positions_listz[i-1]

  print(globals()[f"positions{i}x"])

#위치 가중치 리스트
print('위치 가중치 리스트')

for i in range(1, 7):
    globals()[f"lw{i}"] = globals()[f"positions{i}x"]
    
lw_list = positions_listz

#정신 상태&인지 의미 원소 비율+위치 가중치 리스트
for i in range(1, 7):
    globals()[f"x{i}"] = globals()[f"positions{i}x"] + globals()[f"ppa{i}"]
x_list = [x1, x2, x3, x4, x5, x6]

print(lw_list)
print(sorted(lw_list))
print(x_list)
lw_list = sorted(lw_list)

for i in range(1, 7):
    l = lw_list.index(globals()[f"lw{i}"])
    x_list.remove(globals()[f"x{i}"])
    x_list.insert(l, globals()[f"x{i}"])

#최종 x백터 리스트
print(x_list)

weight_list = [0.8, 0.7, 1.0, 1.0, 0.5, 0.4]
import numpy as np
for i in range(1, 7):
    e = x_list.index(globals()[f"x{i}"])
    x_list[e] = np.array(globals()[f"x{i}"])*(weight_list[i-1])
    (globals()[f"x{i}"]) = (globals()[f"x{i}"])*weight_list[i-1]

    print(x_list)

for i in range(1, 7):
    if (globals()[f"x{i}"]) == 0:
        x_list.remove(globals()[f"x{i}"])
x_list1 = x_list
print('정신 상태&인지')
print(x_list1)

print('--------------------------------------------------------------')


#실체
#I = 1
#YOU = 2
#SOMEONE = 3
#SOMETHING = 4
#THING = 5
#PEOPLE = 6
#BODY = 7


iysstpb_list = [I, YOU, SOMEONE, SOMETHING, SEE, HEAR]

for i in range(1, 7):
    globals()[f"positions{i}"] = [s for s, v in enumerate(A) if v == iysstpb_list[i-1]]
    print(globals()[f"positions{i}"])

for i in range(1, 7):
    if  iysstpb_list[i-1] in A:
      print(0)
      print(A.count(iysstpb_list[i-1]))
      iysstpb_list[i-1] = A.count(iysstpb_list[i-1])

all_positions = positions1 + positions2 + positions3 + positions4 + positions5 + positions6


#정신 상태&인지 의미 원소 비율
for i in range(1, 7):
    if len(all_positions) == 0:
        print(0)
        globals()[f"ppa{i}"] = 0
    else:
        print(len(globals()[f"positions{i}"])/len(all_positions))
        globals()[f"ppa{i}"] = len(globals()[f"positions{i}"])/len(all_positions)


#positions1x = sum(positions1)/total_count
#위치 가중치
print('위치 가중치')
positions_listz = [0]*6
positions_list = [positions1, positions2, positions3, positions4, positions5, positions6]

for i in range(1, 7):
  positions_listz[i-1] = sum(positions_list[i-1])/total_count
  globals()[f"positions{i}x"] = positions_listz[i-1]

  print(globals()[f"positions{i}x"])

#위치 가중치 리스트
print('위치 가중치 리스트')

for i in range(1, 7):
    globals()[f"lw{i}"] = globals()[f"positions{i}x"]
    
lw_list = positions_listz

#정신 상태&인지 의미 원소 비율+위치 가중치 리스트
for i in range(1, 7):
    globals()[f"x{i}"] = globals()[f"positions{i}x"] + globals()[f"ppa{i}"]
x_list = [x1, x2, x3, x4, x5, x6]

print(lw_list)
print(sorted(lw_list))
print(x_list)
lw_list = sorted(lw_list)

for i in range(1, 7):
    l = lw_list.index(globals()[f"lw{i}"])
    x_list.remove(globals()[f"x{i}"])
    x_list.insert(l, globals()[f"x{i}"])

#최종 x백터 리스트
print(x_list)

weight_list = [0.8, 0.7, 1.0, 1.0, 0.5, 0.4]
import numpy as np
for i in range(1, 7):
    e = x_list.index(globals()[f"x{i}"])
    x_list[e] = np.array(globals()[f"x{i}"])*(weight_list[i-1])
    (globals()[f"x{i}"]) = (globals()[f"x{i}"])*weight_list[i-1]

    print(x_list)

for i in range(1, 7):
    if (globals()[f"x{i}"]) == 0:
        x_list.remove(globals()[f"x{i}"])

x_list2 = x_list
print('실체')
print(x_list2)








print('--------------------------------------------------------------')

#combined_list = [val for pair in zip(Sentence_order_list1, Sentence_order_list) for val in pair]
#print(combined_list)

#여기서 부터
from itertools import zip_longest

combined_list = [
    val
    for pair in zip_longest(x_list1, x_list2)
    for val in pair
    if val is not None
]

print(combined_list)
