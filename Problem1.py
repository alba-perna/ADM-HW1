# INTRODUCTION
# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")


# Python If-Else
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n%2 == 1:
        print("Weird")
    elif n >= 2 and n <= 5:
        print("Not Weird")
    elif n >= 6 and n <= 20:
        print("Weird")
    elif n > 20:
        print("Not Weird")


# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range (0, n):
        print(i**2)


# Write a function
def is_leap(year):
    leap = False
    if year%4 == 0:
        leap = True
        if year%100 == 0 and year%400!=0:
            leap = False
    return leap

year = int(input())
print(is_leap(year))


# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range (0, n):
        print(i+1, end="")


# DATA TYPES
# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    list = [[i, j, k] for i in range(x + 1) for j in range(y + 1)
    for k in range(z + 1) if i + j + k != n]
    print(list)


# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    scores = sorted(set(arr), reverse=True)
    print(scores[1])


# Nested Lists
if __name__ == '__main__':
    records = []
    scores = []
    names = []

    for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name, score])
        scores.append(score)

    second_lowest = sorted(set(scores))[1]

    for record in records:
        if record[1] == second_lowest:
            names.append(record[0])
    for name in sorted(names):
        print(name)


# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    sum = 0

    for score in student_marks[query_name]:
        sum += score
    print("{:.2f}".format(sum/3))


# Lists
if __name__ == '__main__':
    N = int(input())
    list = []
    for i in range(N):
        command=input().split()
        if command[0] == "insert":
            list.insert(int(command[1]), int(command[2]))
        elif command[0] == "print":
            print(list)
        elif command[0] == "remove":
            list.remove(int(command[1]))
        elif command[0] == "append":
            list.append(int(command[1]))
        elif command[0] == "sort":
            list.sort()
        elif command[0] == "pop":
            list.pop()
        elif command[0] == "reverse":
            list.reverse()


# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))


# STRINGS
# sWAP cASE
def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join
def split_and_join(line):
    a = line.split(" ")
    return "-".join(a)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# What's Your Name?
def print_full_name(first, last):
    print("Hello " + first + " " + last + "! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(0, len(string)):
        if string[i:len(sub_string)+i] == sub_string:
            count += 1
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)


# String Validators
if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))


# Text Alignment
thickness = int(input())
c = 'H'
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# Text Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# Designer Door Mat
if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range (n//2):
        j = 2*i+1
        print((j*'.|.').center(m, '-'))
    print(('WELCOME').center(m, '-'))
    for i in reversed (range (n//2)):
        j = 2*i+1
        print((j*'.|.').center(m, '-'))


# String Formatting
def print_formatted(number):
    size = len(format(number, 'b'))
    for i in range (1, number+1):
        octal = format(i, 'o')
        hexa = format(i, 'X')
        binary = format(i, 'b')
        print(str(i).rjust(size), str(octal).rjust(size), str(hexa).rjust(size), str(binary).rjust(size))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# Alphabet Rangoli
import string

def print_rangoli(size):
    nbin = 4*n-3
    alpha = string.ascii_lowercase
    lines = []
    for i in range (size):
        row = "-".join(alpha[i:size])
        lines.append((row[::-1]+row[1:]).center(nbin, "-"))
    print('\n'.join(lines[:0:-1]+lines))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!
import math
import os
import random
import re
import sys

def solve(s):
    name = (word.capitalize() for word in s.split(' '))
    return ' '.join(name)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()


# The Minion Game
def minion_game(string):
    k = 0
    s = 0
    for i in range (len(string)):
        char = string[i]
        if char in ['A', 'E', 'I', 'O', 'U']:
            k += len(string)-i
        else:
            s += len(string)-i
    if k > s:
        print('Kevin', k)
    elif s > k:
        print('Stuart', s)
    else:
        print('Draw')

if __name__ == '__main__':
    s = input()
    minion_game(s)


# Merge the Tools!
def merge_the_tools(string, k):
    for i in range (len(string)//k):
        seq = ''
        s = string[0:k]
        for char in s:
            if char not in seq:
                seq += char
        print(seq)
        string = string[k:]

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# SETS
# Introduction to Sets
def average(array):
    return "{:.2f}".format(sum(set(array))/len(set(array)))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# Symmetric Difference
def symmetric_difference(m, n):
    a = m.difference(n)
    b = n.difference(m)
    return sorted(a.union(b))

if __name__ == '__main__':
    m = int(input())
    m_arr = set(map(int, input().split()))
    n = int(input())
    n_arr = set(map(int, input().split()))
    diff = symmetric_difference(m_arr, n_arr)
    for num in diff:
        print(num)

# No Idea!
def happiness(arr, A, B):
    sum = 0
    for i in arr:
        if (i in A) == True:
            sum += 1
        elif (i in B) == True:
            sum -=1
    return sum

if __name__ == '__main__':
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    A = set(map(int, input().split()))
    B = set(map(int, input().split()))
    total = happiness(arr, A, B)
    print(total)


# Set .add()
if __name__ == '__main__':
    N = int(input())
    countries = set()
    for i in range (N):
        c = input()
        countries.add(c)
    print(len(countries))


# Set .discard(), .remove() & .pop()
if __name__ == '__main__':
    n = int(input())
    arr = set(map(int, input().split()))
    N = int(input())
    for i in range(N):
        command = list(map(str, input().split()))
        if command[0] == "remove":
            if (int(command[1]) in arr) == True:
                arr.remove(int(command[1]))
        elif command[0] == "discard":
            arr.discard(int(command[1]))
        elif command[0] == "pop":
            if (len(arr) > 0):
                arr.pop()
    print(sum(arr))

# Set .union() Operation
if __name__ == '__main__':
    n = int(input())
    n_arr = set(map(int, input().split()))
    b = int(input())
    b_arr = set(map(int, input().split()))
    print(len(n_arr.union(b_arr)))


# Set .intersection() Operation
if __name__ == '__main__':
    n = int(input())
    n_arr = set(map(int, input().split()))
    b = int(input())
    b_arr = set(map(int, input().split()))
    print(len(n_arr.intersection(b_arr)))


# Set .difference() Operation
if __name__ == '__main__':
    n = int(input())
    n_arr = set(map(int, input().split()))
    b = int(input())
    b_arr = set(map(int, input().split()))
    print(len(n_arr.difference(b_arr)))


# Set .symmetric_difference() Operation
if __name__ == '__main__':
    n = int(input())
    n_arr = set(map(int, input().split()))
    b = int(input())
    b_arr = set(map(int, input().split()))
    print(len(n_arr.symmetric_difference(b_arr)))


# Set Mutations
if __name__ == '__main__':
    A = int(input())
    arr = set(map(int, input().split()))
    N = int(input())
    for i in range (N):
        command = list(map(str, input().split()))
        if command[0] == "update":
            s = set(map(int, input().split()))
            arr.update(s)
        elif command[0] == "intersection_update":
            s = set(map(int, input().split()))
            arr.intersection_update(s)
        elif command[0] == "difference_update":
            s = set(map(int, input().split()))
            arr.difference_update(s)
        elif command[0] == "symmetric_difference_update":
            s = set(map(int, input().split()))
            arr.symmetric_difference_update(s)
    print(sum(arr))


# The Captain's Room
if __name__ == '__main__':
    k = int(input())
    rooms = sorted(map(int, input().split()))
    for i in range(len(rooms)):
        if(i != len(rooms)-1):
            if(rooms[i]!=rooms[i-1] and rooms[i]!=rooms[i+1]):
                print(rooms[i])
                break;
        else:
            print(rooms[i])


# Check Subset
if __name__ == '__main__':
    t = int(input())
    for i in range (t):
        a = int(input())
        a_set = set(map(int, input().split()))
        b = int(input())
        b_set = set(map(int, input().split()))
        print(a_set.difference(b_set) == set())


# Check Strict Superset
if __name__ == '__main__':
    a_set = set(map(int, input().split()))
    n = int(input())
    is_superset = True
    for i in range(n):
        if is_superset is False:
            break
        n_set = set(map(int, input().split()))
        if not n_set.issubset(a_set) or len(n_set) >= len(a_set):
            is_superset = False
    print(is_superset)


# COLLECTIONS
# collections.Counter()
from collections import Counter

if __name__ == '__main__':
    x = int(input())
    shoes = Counter(map(int, input().split()))
    n = int(input())
    total = 0
    for i in range(n):
        size, price = map(int, input().split())
        if shoes[size]:
            total += price
            shoes[size] -= 1
    print(total)


# DefaultDict Tutorial
from collections import defaultdict

if __name__ == '__main__':
    n, m = map(int, input().split())
    d = defaultdict(list)
    for i in range(n):
        a = input()
        d[a].append(i+1)
    for i in range(m):
        b = input()
        if b in d:
            print(*d[b])
        else:
            print(-1)


# Collections.namedtuple()
from collections import namedtuple

if __name__ == '__main__':
    n = int(input())
    Student = namedtuple('Student', input())

    total = 0
    for i in range(n):
        s = Student(*input().split())
        total += int(s.MARKS)

    print("{:.2f}".format(total/n))


# Collections.OrderedDict()
from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    ordered_dictionary = OrderedDict()
    for i in range(n):
        item = input().split()
        price = int(item[-1])
        name = " ".join(item[:-1])

        if (ordered_dictionary.get(name)):
            ordered_dictionary[name] += int(price)
        else:
            ordered_dictionary[name] = int(price)
    for i in ordered_dictionary.keys():
        print(i, ordered_dictionary[i])


# Word Order
from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    ordered_dictionary = OrderedDict()
    for i in range(n):
        word = input()
        if ordered_dictionary.get(word):
            ordered_dictionary[word] += 1
        else:
            ordered_dictionary[word] = 1
    print(len(ordered_dictionary))
    for key, value in ordered_dictionary.items():
        print(value,  end=" ")


# Collections.deque()
from collections import deque

if __name__ == '__main__':
    N = int(input())
    d = deque()
    for i in range(N):
        command=input().split()
        if command[0] == "append":
            d.append(command[1])
        elif command[0] == "pop":
            d.pop()
        elif command[0] == "popleft":
            d.popleft()
        elif command[0] == "appendleft":
            d.appendleft(command[1])
    for i in d:
        print(i, end=" ")


# Piling Up!
from collections import deque

if __name__ == '__main__':
    t = int(input())
    for i in range(t):
        n = int(input())
        d = deque(map(int, input().split()))
        res = "Yes"
        length = 2**31
        for i in range(n):
            if d[0] >= d[-1] and d[0] <= length:
                length = d[0]
                d.popleft()
            elif d[0] < d[-1] and d[-1] <= length:
                length = d[-1]
                d.pop()
            else:
                res = "No"
                break
        print(res)


# Company Logo
import math
import os
import random
import re
import sys
from collections import Counter

if __name__ == '__main__':
    s = input()
    lis = Counter(list(sorted(s)))
    for key, value in lis.most_common(3):
        print(key,value)


# DATE AND TIME
# Calendar Module
import calendar

if __name__ == '__main__':
    date = input().split()
    day = calendar.weekday(int(date[2]), int(date[0]), int(date[1]))
    name = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
    print(name[day])


# Time Delta
import math
import os
import random
import re
import sys

from datetime import datetime

def time_delta(t1, t2):
    t1 = datetime.strptime(t1,"%a %d %b %Y %H:%M:%S %z")
    t2 = datetime.strptime(t2,"%a %d %b %Y %H:%M:%S %z")
    return str(int(abs((t1-t2).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()


# EXCEPTIONS
if __name__ == '__main__':
    t = int(input())
    for i in range(t):
        a, b = input().split()
        try:
            print(int(a)//int(b))
        except ZeroDivisionError as e1:
            print("Error Code:", e1)
        except ValueError as e2:
            print("Error Code:", e2)


# BUILT-INS
# Zipped!
if __name__ == '__main__':
    n, x = map(int, input().split())
    marks = list()
    for i in range(x):
        m = map(float, input().split())
        marks.append(m)
    for i in zip(*marks):
        print("{:.1f}".format(sum(i)/x))


# Athlete Sort
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())

    arr.sort(key=lambda x: x[k])
    for i in arr:
        print(*i, sep=" ")


# ginortS
if __name__ == '__main__':
    word = input()
    lower=[]
    upper=[]
    odd=[]
    even=[]
    for char in sorted(word):
        if char.isalpha():
            if char.islower():
                lower.append(char)
            else:
                upper.append(char)
        elif int(char)%2==0:
            even.append(char)
        else:
            odd.append(char)

    print(*lower, *upper, *odd, *even, sep="")


# PYTHON FUNCTIONALS
cube = lambda x: x**3

def fibonacci(n):
    if n==0:
        return list()
    fib = [0,1]
    for i in range(2,n):
        fib.append(fib[i-2] + fib[i-1])

    return(fib[0:n])

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# REGEX AND PARSING CHALLENGES
# Detect Floating Point Number
import re

if __name__ == '__main__':
    t = int(input())
    for i in range(t):
        n = input()
        float = re.compile(r'^[+-]?[0-9]*\.[0-9]+$')
        if re.fullmatch(float, n):
            print("True")
        else:
            print("False")


# Re.split()
regex_pattern = r"[,.]"

import re
print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()
import re
s = re.search(r'([a-zA-Z0-9])\1', input().strip())
print(s.group(1) if s else -1)


# Re.findall() & Re.finditer()
import re
s = re.findall(r'(?<=[^aeiouAEIOU ])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU ])', input())
if s:
    for i in s:
        print(i)
else:
    print(-1)


# Re.start() & Re.end()
import re
s, k = input(), input()
matches = re.finditer(r'(?=(' + k + '))', s)
is_match = False
for match in matches:
    is_match = True
    print((match.start(1), match.end(1) - 1))
if is_match == False:
    print((-1, -1))


# Regex Substitution
import re
n = int(input())
for i in range(n):
    s = re.sub("(?<=\s)&&(?=\s)", "and", input())
    print(re.sub("(?<=\s)\|\|(?=\s)", "or", s))


# Validating Roman Numerals
digits = '(V?[I]{0,3}|I[VX])'
tens = '(L?[X]{0,3}|X[LC])'
hundreds = '(D?[C]{0,3}|C[DM])'
thousands = 'M{0,3}'
regex_pattern = r"%s%s%s%s$" % (thousands, hundreds, tens, digits)

import re
print(str(bool(re.match(regex_pattern, input()))))


# Validating phone numbers
import re
n = int(input())
for i in range(n):
    number = input()
    if(len(number)==10 and number.isdigit()):
        output = re.findall(r"^[789]\d{9}$",number)
        if(len(output)==1):
            print("YES")
        else:
            print("NO")
    else:
        print("NO")


# Validating and Parsing Email Addresses
import re
n = int(input())
for i in range(n):
    name, email = input().split()
    pattern="<[a-z][a-zA-Z0-9\-\.\_]+@[a-zA-Z]+\.[a-zA-Z]{1,3}>"
    if bool(re.match(pattern, email)):
        print(name, email)


# Hex Color Code
import re
n = int(input())
css = False
for i in range(n):
    s = input()
    if '{' in s:
        css = True
    elif '}' in s:
        css = False
    elif css:
        for color in re.findall('#[0-9a-fA-F]{3,6}', s):
            print(color)


# HTML Parser - Part 1
from html.parser import HTMLParser

class HTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for e in attrs:
            print ('->', e[0], '>', e[1])

    def handle_endtag(self, tag):
        print ('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for e in attrs:
            print ('->', e[0], '>', e[1])

parser = HTMLParser()
n = int(input())
for i in range(n):
    html = input()
    parser.feed(html)


# HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if (len(data.split('\n')) != 1):
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.replace("\r", "\n"))

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr, value in attrs:
            print("->", attr, ">", value)

    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr, value in attrs:
            print("->", attr, ">", value)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Validating UID
import re

for i in range(int(input())):
    n = input().strip()
    if n.isalnum() and len(n) == 10:
        if bool(re.search(r'(.*[A-Z]){2,}',n)) and bool(re.search(r'(.*[0-9]){3,}',n)):
            if re.search(r'.*(.).*\1+.*',n):
                print('Invalid')
            else:
                print('Valid')
        else:
            print('Invalid')
    else:
        print('Invalid')


# Validating Credit Card Numbers
import re

pattern = re.compile(
    r'^'
    r'(?!.*(\d)(-?\1){3})'
    r'[456]\d{3}'
    r'(?:-?\d{4}){3}'
    r'$')
n = int(input())
for i in range(n):
    print('Valid' if pattern.search(input().strip()) else 'Invalid')


# Validating Postal Codes
regex_integer_in_range = r"^[1-9][\d]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

import re
P = input()
print (bool(re.match(regex_integer_in_range, P))
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# Matrix Script
import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
matrix = list(zip(*matrix))
sample = str()

for row in matrix:
    for char in row:
        sample += char
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', sample))


# XML
# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    score = 0
    for tag in node:
        score = score + get_attr_number(tag)
    return score + len(node.attrib)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

max = 0
def depth(e, level):
    global max
    if (level == max):
        max += 1
    for x in e:
        depth(x, level + 1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(max)


# CLOSURES AND DECORATIONS
# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(['+91 ' + x[-10:-5] + ' ' + x[-5:] for x in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# NUMPY
# Arrays
import numpy

def arrays(arr):
    return(numpy.array(arr[::-1], float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape
import numpy

arr = numpy.array(input().split(' '), int)
print(arr.reshape(3, 3))


# Transpose and Flatten
import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(arr.transpose())
print(arr.flatten())


# Concatenate
import numpy

n, m, p = map(int, input().split())
array_1 = numpy.array([input().split() for i in range(n)], int)
array_2 = numpy.array([input().split() for i in range(m)], int)
print(numpy.concatenate((array_1, array_2), axis=0))


# Zeros and Ones
import numpy

n = tuple(map(int, input().split()))
print(numpy.zeros(n, int))
print(numpy.ones(n, int))


# Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')

n,m = map(int, input().split())
print(numpy.eye(n, m))


# Array Mathematics
import numpy

n, m = map(int, input().split())
A = numpy.array([input().split() for i in range(n)], int)
B = numpy.array([input().split() for i in range(n)], int)

print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)


# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')

arr = numpy.array(input().split(),float)
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))


# Sum and Prod
import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(numpy.prod(numpy.sum(arr, axis=0), axis=0))


# Min and Max
import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(numpy.max(numpy.min(arr, axis=1), axis=0))


# Mean, Var, and Std
import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(numpy.round(numpy.std(arr), 11))


# Dot and Cross
import numpy

n = int(input())
A = numpy.array([input().split() for i in range(n)], int)
B = numpy.array([input().split() for i in range(n)], int)
print(numpy.dot(A, B))


# Inner and Outer
import numpy

A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
print(numpy.inner(A, B), numpy.outer(A, B), sep='\n')


# Polynomials
import numpy

coef = list(map(float, input().split()))
coef.reverse()
x = int(input())
value = 0
for i in range(len(coef)):
    value += coef[i]*pow(x, i)
print(value)


# Linear Algebra
import numpy

n = int(input())
arr = numpy.array([input().split() for i in range(n)], float)
print(round(numpy.linalg.det(arr),2))
