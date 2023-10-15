# Birthday Cake Candles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    candles = sorted(candles, reverse=True)
    max = candles[0]
    count = 0
    for i in range(len(candles)):
        if candles[i] == max:
            count += 1
        else:
            break
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()


# Kangaroo: Number Line Jumps
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v1 <= v2:
        return "NO"
    elif (x1-x2) % (v2-v1) == 0:
        return 'YES'
    else:
        return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()


# Strange advertising
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    advertised = 5
    total = 0
    for i in range(n):
        liked = advertised//2
        total += liked
        advertised = liked*3
    return total

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()


# Recursive digit sum
import math
import os
import random
import re
import sys

def superDigit(n, k):
    p = k * sum(int(x) for x in n)
    return sumDigit(str(p))

def sumDigit(n):
    if int(n) < 10:
        return n
    return sumDigit(str(sum(int(x) for x in n)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()


# Insertion Sort 1
import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    num = arr[n-1]
    idx = n-2
    while idx >= 0 and num < arr[idx]:
        arr[idx+1] = arr[idx]
        print(str(arr)[1:-1].replace(",", ""))
        idx -= 1
    arr[idx+1] = num
    print(str(arr)[1:-1].replace(",", ""))

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)


# Insertion Sort 2
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(n):
        idx = i
        while (idx > 0 and arr[idx] < arr[idx-1]):
            num = arr[idx]
            arr[idx] = arr[idx-1]
            arr[idx-1] = num
            idx -= 1
        if (i != 0):
            print(str(arr)[1:-1].replace(",", ""))

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)
