

f = open('convertcsv.csv', mode='r', encoding='utf-8')
for line in f:
    character = line.split(',')[1]
    print(character, end='')