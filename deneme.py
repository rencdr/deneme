#Bubble Sort
def bubble_sort(arr):
  n = len(arr)

  for i in range(n - 1):
      for j in range(0, n - i - 1):
          if arr[j] > arr[j + 1]:
              arr[j], arr[j + 1] = arr[j + 1], arr[j]

# bkz
my_list = [5, 2, 8, 12, 3]
bubble_sort(my_list)
print("Sıralanmış liste:", my_list)