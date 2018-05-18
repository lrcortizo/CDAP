def split_fileBook(line):
    (book,store) = line.split(',')
    if store == libreria:
        return (book, store)
    else:
        return ("", "")

def split_fileNum(line):
    (book,count) = line.split(',')
    count = int(count)
    return (book,count)

def sum_counts(a, b):
    return a + b

def main(sc, libreria):
    # Load the books lookup dictionary
    books = sc.textFile("join2_book*.txt").map(split_fileBook)
    # Broadcast the lookup dictionary to the cluster
    #book_lookup = sc.broadcast(books)

    # Load the nums lookup dictionary
    nums = sc.textFile("join2_num*.txt").map(split_fileNum)
    nums_joined_books = nums.join(books)

    nums_joined_books = nums_joined_books.reduceByKey(sum_counts).collect()

    total = 0
    f = open('output.txt', 'w')
    for entry in nums_joined_books:
        print "%s: %s" % (entry[0], entry[1][0])
        f.write("%s: %s\n" % (entry[0], entry[1][0]))
        total = total + entry[1][0]
    print "TOTAL: %s" % (total)
    f.write("TOTAL: %s" % (total))

    f.close()

# Configure Spark

from pyspark import SparkContext
sc = SparkContext(appName="novelasLibrerias")
sc.setLogLevel("WARN")
import sys
libreria = sys.argv[1]
print ("Obteniendo ventas de novelas ofrecidas por "+libreria)

# Execute Main functionality
main(sc, libreria)
