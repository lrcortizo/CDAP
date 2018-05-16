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

    # Map the total delay to the airline (joined using the broadcast value)
    #counts = nums.map(lambda b: (book_lookup.value[b.book], b.count))
    # Reduce the total delay for the month to the airline
    #counts = counts.reduceByKey(sum_counts).collect()
    #counts = sorted(counts, key=itemgetter(1))
    total = 0
    for entry in nums_joined_books:
        print "%s: %s" % (entry[0], entry[1][0])
        total = total + entry[1][0]
            # Show a bar chart of the delays
            #plot(delays)
    print "TOTAL: %s" % (total)

# Configure Spark

from pyspark import SparkContext
sc = SparkContext(appName="novelasLibrerias")
sc.setLogLevel("WARN")
import sys
libreria = sys.argv[1]
print ("Obteniendo ventas de novelas ofrecidas por "+libreria)

# Execute Main functionality
main(sc, libreria)
