def split_fileBook(line):
    (book,store) = line.split(',')
    if store == libreria:
        return (book, 0)
    else:
        return ("", "")

def split_fileNum(line):
    (book,count) = line.split(',')
    count = int(count)
    return (book,count)

def sum_counts(a, b):
    return a + b

def main(sc):
    # Se carga el diccionario de books
    books = sc.textFile("join2_book*.txt").map(split_fileBook)

    # Se carga el diccionario de nums
    nums = sc.textFile("join2_num*.txt").map(split_fileNum)
    #Se hace el join de ambas tuplas a traves del atributo comun 'book'
    nums_joined_books = nums.join(books)

    #Se hace el reduce acumulando la suma de libros vendidos para los que tienen el mismo nombre
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

if len(sys.argv) != 2:
    sys.exit("Es necesario especificar una libreria")

libreria = sys.argv[1]
print ("Obteniendo ventas de novelas ofrecidas por "+libreria)

# Execute Main functionality
main(sc)
