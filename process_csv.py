import csv
import io
p = open('tweet.pos', 'a')
n = open('tweet.neg', 'a')
with io.open('training.csv', 'rt', encoding='latin-1') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            if row[0] == '0':
                n.write(row[5] + '\n')
            else:
                p.write(row[5] + '\n')
        except:
            pass
p.close()
n.close()
