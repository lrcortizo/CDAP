def split_fileNum(line):
    (book,count) = line.split(',')
    count = int(count)
    return (book,count)

def split_fileBook(line):
    (book,store) = line.split(',')
    return (book, store)

def sum_counts(a, b):
    return a + b
