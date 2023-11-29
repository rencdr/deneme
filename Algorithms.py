#Algorithms
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

#Insertion Sort 
def insertion_sort(arr):
  for i in range(1, len(arr)):
      key = arr[i]
      j = i - 1

      while j >= 0 and arr[j] > key:
          arr[j + 1] = arr[j]
          j -= 1

      arr[j + 1] = key

# bkz
my_list = [5, 2, 8, 12, 3]
insertion_sort(my_list)
print("Sıralanmış liste:", my_list)

#Selection Sort
def selection_sort(arr):
  for i in range(len(arr)):
      min_index = i

      for j in range(i + 1, len(arr)):
          if arr[j] < arr[min_index]:
              min_index = j

      arr[i], arr[min_index] = arr[min_index], arr[i]

# bkz
my_list = [5, 2, 8, 12, 3]
selection_sort(my_list)
print("Sıralanmış liste:", my_list)

#Quick Sort
def quick_sort(arr):
  if len(arr) <= 1:
      return arr
  else:
      pivot = arr[0]
      less = [x for x in arr[1:] if x <= pivot]
      greater = [x for x in arr[1:] if x > pivot]
      return quick_sort(less) + [pivot] + quick_sort(greater)

# bkz
my_list = [5, 2, 8, 12, 3]
sorted_list = quick_sort(my_list)
print("Sıralanmış liste:", sorted_list)

#Heap Sort
def heapify(arr, n, i):
  largest = i  # Şu anki düğümü en büyük kabul ediyoruz
  left = 2 * i + 1  # Sol çocuk düğümün dizinini hesaplıyoruz
  right = 2 * i + 2  # Sağ çocuk düğümün dizinini hesaplıyoruz

  # Sol çocuk düğüm, en büyük düğümden daha büyükse
  if left < n and arr[left] > arr[largest]:
      largest = left

  # Sağ çocuk düğüm, en büyük düğümden daha büyükse
  if right < n and arr[right] > arr[largest]:
      largest = right

  # En büyük düğüm farklıysa
  if largest != i:
      arr[i], arr[largest] = arr[largest], arr[i]  # Düğümleri değiştir
      heapify(arr, n, largest)  # Alt ağaçları yeniden düzenle

def heap_sort(arr):
  n = len(arr)

  # Max heap oluştur
  for i in range(n // 2 - 1, -1, -1):
      heapify(arr, n, i)

  # Heap'den elemanları çıkararak sırala
  for i in range(n - 1, 0, -1):
      arr[i], arr[0] = arr[0], arr[i]  # En büyük elemanı sıralı kısma taşı
      heapify(arr, i, 0)  # Yeniden düzenle

# bkz
my_list = [5, 2, 8, 12, 3]
heap_sort(my_list)
print("Sıralanmış liste:", my_list)

