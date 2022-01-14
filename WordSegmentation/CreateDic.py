import pickle

with open('VNDictionary.txt', 'r', encoding= 'utf-8') as f:
  text = f.readlines()
b = []
for i in text:
  if i[0] == '#':
    i = i[2::]
    a = i.split()
    b.append('_'.join(a))
pickle.dump(b, open('VietNamDictionary.pkl','wb'))