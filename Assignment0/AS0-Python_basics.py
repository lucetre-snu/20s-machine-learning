#!/usr/bin/env python
# coding: utf-8

# # M2608.001300 기계학습 기초 및 전기정보 응용<br> Assignment 0: Python Basics

# ## Problem 1: Selection Sort
# 
# 아래 selection sort 함수를 구현해보세요. 
# YOUR CODE COMES HERE 라는 주석이 있는 곳을 채우면 됩니다.

# In[94]:


def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        minIndex = i
        for j in range(i+1, n):
            if arr[minIndex] > arr[j]:
                minIndex = j
        arr[minIndex], arr[i] = arr[i], arr[minIndex]
#         print('step {}'.format(i+1), arr)
    
    return arr


# In[95]:


import random
array = [random.randint(0, 20) for _ in range(20)]
print('array: ', array)

array_sorted = selection_sort(array)
print('sorted:', array_sorted)

print()
print('Q: Is the array sorted?')
print('A:', sorted(array) == array_sorted)


# ## Problem 2: Classes
# 
# Selection sort, insertion sort, merge sort를 아래 class의 instance method로 구현해 보세요. <br>
# YOUR CODE COMES HERE 라는 주석이 있는 곳을 채우면 됩니다.

# In[96]:


class Sorter:
    def __init__(self, method):
        self.method = method
        
    @staticmethod
    def of(method):
        return Sorter(method)
        
    def sort(self, arr):
        if self.method == 'selection_sort':
            return self.selection_sort(arr)
        
        elif self.method == 'insertion_sort':
            return self.insertion_sort(arr)
        
        elif self.method == 'merge_sort':
            return self.merge_sort(arr)
        
        else:
            raise ValueError('Unknown method: %s' % method)

    def selection_sort(self, arr):
        # not to affect the original arr
        arr = arr[:]
        n = len(arr)
        
        # selection sort
        for i in range(n-1):
            minIndex = i
            for j in range(i+1, n):
                if arr[minIndex] > arr[j]:
                    minIndex = j
            arr[minIndex], arr[i] = arr[i], arr[minIndex]
#             print('step {}'.format(i+1), arr)
        return arr
    
    def insertion_sort(self, arr):
        # not to affect the original arr
        arr = arr[:]
        n = len(arr)
        
        # insertion sort
        for i in range(1, n):
            key_to_insert = arr[i]
            index_to_insert = i
            for j in range(i-1, -1, -1):
                if arr[j] < key_to_insert:
                    break
                arr[j+1] = arr[j]
                index_to_insert = j
            arr[index_to_insert] = key_to_insert;
#             print('step {}'.format(i), arr)
        return arr
    
    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr
        # not to affect the original arr
        arr = arr[:]
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []
        while len(left) > 0 or len(right) > 0:
            if len(left) > 0 and len(right) > 0:
                if left[0] <= right[0]:
                    result.append(left[0])
                    left = left[1:]
                else:
                    result.append(right[0])
                    right = right[1:]
            elif len(left) > 0:
                result.append(left[0])
                left = left[1:]
            elif len(right) > 0:
                result.append(right[0])
                right = right[1:]
        return result


# In[97]:


array = [random.randint(0, 20) for _ in range(20)]

algorithms = ['selection_sort', 'insertion_sort', 'merge_sort']
for algorithm in algorithms:
    sorter = Sorter.of(algorithm)
    array_sorted = sorter.sort(array)
    print('%s sorted? %s' % (algorithm, sorted(array) == array_sorted))


# In[ ]:




