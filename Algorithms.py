# #Algorithms
# #Bubble Sort
# def bubble_sort(arr):
#   n = len(arr)

#   for i in range(n - 1):
#       for j in range(0, n - i - 1):
#           if arr[j] > arr[j + 1]:
#               arr[j], arr[j + 1] = arr[j + 1], arr[j]

# # bkz
# my_list = [5, 2, 8, 12, 3]
# bubble_sort(my_list)
# print("Sıralanmış liste:", my_list)

# #Insertion Sort 
# def insertion_sort(arr):
#   for i in range(1, len(arr)):
#       key = arr[i]
#       j = i - 1

#       while j >= 0 and arr[j] > key:
#           arr[j + 1] = arr[j]
#           j -= 1

#       arr[j + 1] = key

# # bkz
# my_list = [5, 2, 8, 12, 3]
# insertion_sort(my_list)
# print("Sıralanmış liste:", my_list)

# #Selection Sort
# def selection_sort(arr):
#   for i in range(len(arr)):
#       min_index = i

#       for j in range(i + 1, len(arr)):
#           if arr[j] < arr[min_index]:
#               min_index = j

#       arr[i], arr[min_index] = arr[min_index], arr[i]

# # bkz
# my_list = [5, 2, 8, 12, 3]
# selection_sort(my_list)
# print("Sıralanmış liste:", my_list)

# #Quick Sort
# def quick_sort(arr):
#   if len(arr) <= 1:
#       return arr
#   else:
#       pivot = arr[0]
#       less = [x for x in arr[1:] if x <= pivot]
#       greater = [x for x in arr[1:] if x > pivot]
#       return quick_sort(less) + [pivot] + quick_sort(greater)

# # bkz
# my_list = [5, 2, 8, 12, 3]
# sorted_list = quick_sort(my_list)
# print("Sıralanmış liste:", sorted_list)

# #Heap Sort
# def heapify(arr, n, i):
#   largest = i  # Şu anki düğümü en büyük kabul ediyoruz
#   left = 2 * i + 1  # Sol çocuk düğümün dizinini hesaplıyoruz
#   right = 2 * i + 2  # Sağ çocuk düğümün dizinini hesaplıyoruz

#   # Sol çocuk düğüm, en büyük düğümden daha büyükse
#   if left < n and arr[left] > arr[largest]:
#       largest = left

#   # Sağ çocuk düğüm, en büyük düğümden daha büyükse
#   if right < n and arr[right] > arr[largest]:
#       largest = right

#   # En büyük düğüm farklıysa
#   if largest != i:
#       arr[i], arr[largest] = arr[largest], arr[i]  # Düğümleri değiştir
#       heapify(arr, n, largest)  # Alt ağaçları yeniden düzenle

# def heap_sort(arr):
#   n = len(arr)

#   # Max heap oluştur
#   for i in range(n // 2 - 1, -1, -1):
#       heapify(arr, n, i)

#   # Heap'den elemanları çıkararak sırala
#   for i in range(n - 1, 0, -1):
#       arr[i], arr[0] = arr[0], arr[i]  # En büyük elemanı sıralı kısma taşı
#       heapify(arr, i, 0)  # Yeniden düzenle

# # bkz
# my_list = [5, 2, 8, 12, 3]
# heap_sort(my_list)
# print("Sıralanmış liste:", my_list)

#Binary Search
# def binary_search(arr, target):
#     low = 0
#     high = len(arr) - 1

#     while low <= high:
#         mid = (low + high) // 2
#         if arr[mid] == target:
#             return mid
#         elif arr[mid] < target:
#             low = mid + 1
#         else:
#             high = mid - 1

#     return -1

# # bkz
# arr = [2, 5, 7, 12, 19, 23, 27, 35, 42]
# target = 19
# result = binary_search(arr, target)
# if result != -1:
#     print("Hedef değer", target, "dizi içinde", result, ". indiste bulundu.")
# else:
#     print("Hedef değer dizi içinde bulunamadı.")


#Linear Search
# def linear_search(arr, target):
#     for i in range(len(arr)):
#         if arr[i] == target:
#             return i

#     return -1

# # bkz
# arr = [4, 2, 9, 7, 5, 1]
# target = 7
# result = linear_search(arr, target)
# if result != -1:
#     print("Hedef değer", target, "dizi içinde", result, ". indiste bulundu.")
# else:
#     print("Hedef değer dizi içinde bulunamadı.")

#Quick Sort(Divide and Conquer)
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     else:
#         pivot = arr[-1]
#         smaller = [x for x in arr[:-1] if x <= pivot]
#         greater = [x for x in arr[:-1] if x > pivot]
#         return quick_sort(smaller) + [pivot] + quick_sort(greater)

# # bkz
# arr = [9, 3, 5, 1, 2, 7, 6]
# sorted_arr = quick_sort(arr)
# print(sorted_arr)

#Merge Sort(Divide and Conquer)
# def merge_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     else:
#         mid = len(arr) // 2
#         left = arr[:mid]
#         right = arr[mid:]

#         left = merge_sort(left)
#         right = merge_sort(right)

#         return merge(left, right)

# def merge(left, right):
#     result = []
#     i = 0
#     j = 0

#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]:
#             result.append(left[i])
#             i += 1
#         else:
#             result.append(right[j])
#             j += 1

#     result.extend(left[i:])
#     result.extend(right[j:])
#     return result

# # bkz
# arr = [9, 3, 5, 1, 2, 7, 6]
# sorted_arr = merge_sort(arr)
# print(sorted_arr)

#Matrix Multiplication(Divide and Conquer)
# def matrix_multiplication(matrix1, matrix2):
#     rows1 = len(matrix1)
#     cols1 = len(matrix1[0])
#     rows2 = len(matrix2)
#     cols2 = len(matrix2[0])

#     if cols1 != rows2:
#         raise ValueError("Matris boyutları eşlemiyor. Matris çarpımı için uygun boyutlarda matrisler kullanmalısınız.")

#     result = [[0] * cols2 for _ in range(rows1)]

#     for i in range(rows1):
#         for j in range(cols2):
#             for k in range(cols1):
#                 result[i][j] += matrix1[i][k] * matrix2[k][j]

#     return result

# # bkz
# matrix1 = [[1, 2, 3],
#            [4, 5, 6]]
# matrix2 = [[7, 8],
#            [9, 10],
#            [11, 12]]

# result = matrix_multiplication(matrix1, matrix2)
# for row in result:
#     print(row)

#NQueens(Backtracking)
# def is_safe(board, row, col, n):
#     # Aynı sütunda başka vezir var mı?
#     for i in range(row):
#         if board[i][col] == 1:
#             return False

#     # Sol üst çaprazda başka vezir var mı?
#     i = row - 1
#     j = col - 1
#     while i >= 0 and j >= 0:
#         if board[i][j] == 1:
#             return False
#         i -= 1
#         j -= 1

#     # Sağ üst çaprazda başka vezir var mı?
#     i = row - 1
#     j = col + 1
#     while i >= 0 and j < n:
#         if board[i][j] == 1:
#             return False
#         i -= 1
#         j += 1

#     return True


# def solve_n_queens_util(board, row, n, solutions):
#     if row == n:
#         # Bir çözüm bulundu, tahtayı solutions listesine ekleyelim
#         solution = []
#         for i in range(n):
#             row_str = ""
#             for j in range(n):
#                 if board[i][j] == 1:
#                     row_str += "Q"
#                 else:
#                     row_str += "."
#             solution.append(row_str)
#         solutions.append(solution)
#         return

#     for col in range(n):
#         if is_safe(board, row, col, n):
#             board[row][col] = 1
#             solve_n_queens_util(board, row + 1, n, solutions)
#             board[row][col] = 0


# def solve_n_queens(n):
#     board = [[0] * n for _ in range(n)]
#     solutions = []
#     solve_n_queens_util(board, 0, n, solutions)
#     return solutions


# # bkz
# n = 8
# solutions = solve_n_queens(n)
# print(f"{n} Queens için {len(solutions)} çözüm bulundu:")
# for solution in solutions:
#     for row in solution:
#         print(row)
#     print()

#Rat in Maze(Backtracking)
# def solve_maze(maze):
#     # Labirentin boyutunu alalım
#     rows = len(maze)
#     cols = len(maze[0])

#     # Çözüm matrisini oluşturalım ve tüm hücreleri 0 ile başlatalım
#     sol_matrix = [[0] * cols for _ in range(rows)]

#     # Rat in a Maze algoritması için yardımcı fonksiyon
#     def solve_maze_util(row, col):
#         # Hedefe ulaşıldıysa, çözüm bulundu
#         if row == rows - 1 and col == cols - 1:
#             sol_matrix[row][col] = 1
#             return True

#         # Geçerli hücreyi işaretleyelim
#         sol_matrix[row][col] = 1

#         # Sağa hareket etme
#         if col + 1 < cols and maze[row][col + 1] == 1 and solve_maze_util(row, col + 1):
#             return True

#         # Aşağı hareket etme
#         if row + 1 < rows and maze[row + 1][col] == 1 and solve_maze_util(row + 1, col):
#             return True

#         # Geriye doğru giderek diğer yönlere bakma
#         sol_matrix[row][col] = 0
#         return False

#     # solve_maze_util fonksiyonunu başlangıç noktasında çağırarak labirenti çözelim
#     if solve_maze_util(0, 0):
#         return sol_matrix
#     else:
#         return "Çözüm bulunamadı."


# # bkz
# maze = [
#     [1, 0, 0, 0],
#     [1, 1, 0, 1],
#     [0, 1, 0, 0],
#     [1, 1, 1, 1]
# ]

# solution = solve_maze(maze)
# if isinstance(solution, str):
#     print(solution)
# else:
#     print("Labirentin Çözümü:")
#     for row in solution:
#         print(row)

#Job Sequencing Problem(Greedy)
# def job_sequencing(jobs):
#     # İşleri karlarına göre azalan sırada sıralayalım
#     jobs.sort(key=lambda x: x[1], reverse=True)

#     # Boş zaman çizelgesi oluşturalım
#     schedule = [None] * len(jobs)

#     # İşleri yerleştirme ve maksimum karı hesaplama
#     total_profit = 0
#     for job in jobs:
#         deadline = job[2] - 1  # Son teslim tarihi 0 tabanlı indekslendiği için 1 çıkarıyoruz
#         while deadline >= 0:
#             if schedule[deadline] is None:
#                 schedule[deadline] = job[0]  # İşi bu zaman dilimine yerleştir
#                 total_profit += job[1]  # Karı toplama
#                 break
#             deadline -= 1

#     return schedule, total_profit


# # bkz
# jobs = [
#     ('A', 100, 2),
#     ('B', 19, 1),
#     ('C', 27, 2),
#     ('D', 25, 1),
#     ('E', 15, 3)
# ]

# schedule, total_profit = job_sequencing(jobs)
# print("Yerleştirilen İşler:", schedule)
# print("Toplam Kar:", total_profit)

#0/1 Knapsack Problem(Dynamic Programming)
# def knapsack_01(items, capacity):
#     n = len(items)
#     table = [[0] * (capacity + 1) for _ in range(n + 1)]

#     for i in range(1, n + 1):
#         weight, value = items[i - 1]
#         for j in range(1, capacity + 1):
#             if weight <= j:
#                 table[i][j] = max(value + table[i - 1][j - weight], table[i - 1][j])
#             else:
#                 table[i][j] = table[i - 1][j]

#     max_value = table[n][capacity]

#     selected_items = []
#     j = capacity
#     for i in range(n, 0, -1):
#         if table[i][j] != table[i - 1][j]:
#             selected_items.append(items[i - 1])
#             j -= items[i - 1][0]

#     return selected_items, max_value


# # bkz
# items = [(2, 6), (2, 10), (3, 12)]
# capacity = 5

# selected_items, max_value = knapsack_01(items, capacity)
# print("Çantaya Eklenen Nesneler:", selected_items)
# print("Maksimum Toplam Değer:", max_value)

#Longest Palindromic Subsequence (LPS)(Dynamic Programming)
# def longest_palindromic_subsequence(s):
#     n = len(s)
#     table = [[0] * n for _ in range(n)]

#     for i in range(n):
#         table[i][i] = 1

#     for cl in range(2, n + 1):
#         for i in range(n - cl + 1):
#             j = i + cl - 1
#             if s[i] == s[j]:
#                 table[i][j] = table[i + 1][j - 1] + 2
#             else:
#                 table[i][j] = max(table[i + 1][j], table[i][j - 1])

#     lps_length = table[0][n - 1]

#     lps = []
#     i, j = 0, n - 1
#     while i < j:
#         if s[i] == s[j]:
#             lps.append(s[i])
#             i += 1
#             j -= 1
#         elif table[i][j] == table[i + 1][j]:
#             i += 1
#         else:
#             j -= 1

#     if i == j:
#         lps.append(s[i])

#     return lps, lps_length


# # bkz
# string = "character"
# lps, lps_length = longest_palindromic_subsequence(string)
# print("En Uzun Palindromik Alt Dizi:", lps)
# print("Uzunluk:", lps_length)

#Breadth First Search (BFS) (Graph)
# from collections import deque

# def bfs(graph, start_node):
#     visited = set()
#     queue = deque()

#     queue.append(start_node)
#     visited.add(start_node)

#     while queue:
#         current_node = queue.popleft()
#         print(current_node)  # Düğümü ziyaret et veya başka bir işlem yap

#         for neighbor in graph[current_node]:
#             if neighbor not in visited:
#                 queue.append(neighbor)
#                 visited.add(neighbor)


# # bkz
# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }

# start_node = 'A'
# bfs(graph, start_node)

#Depth First Traversal (or DFS)(Graph)
# def dfs(graph, start_node):
#     visited = set()
#     stack = []

#     stack.append(start_node)

#     while stack:
#         current_node = stack.pop()
#         if current_node not in visited:
#             visited.add(current_node)
#             print(current_node)  # Düğümü ziyaret et veya başka bir işlem yap

#             for neighbor in graph[current_node]:
#                 stack.append(neighbor)


# # bkz
# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }

# start_node = 'A'
# dfs(graph, start_node)

#Bellman–Ford (Graph)
# def bellman_ford(graph, start_node):
#     distance = {node: float('inf') for node in graph}
#     distance[start_node] = 0

#     for _ in range(len(graph) - 1):
#         for node in graph:
#             for neighbor, weight in graph[node].items():
#                 if distance[node] + weight < distance[neighbor]:
#                     distance[neighbor] = distance[node] + weight

#     # Negatif döngü kontrolü
#     for node in graph:
#         for neighbor, weight in graph[node].items():
#             if distance[node] + weight < distance[neighbor]:
#                 raise ValueError("Negatif döngü bulundu!")

#     return distance


# # bkz
# graph = {
#     'A': {'B': -1, 'C': 4},
#     'B': {'C': 3, 'D': 2, 'E': 2},
#     'C': {},
#     'D': {'B': 1, 'C': 5},
#     'E': {'D': -3}
# }

# start_node = 'A'
# distances = bellman_ford(graph, start_node)

# for node, distance in distances.items():
#     print(f"{node}: {distance}")

#Kruskal’s Minimum Spanning Tree (MST) (Graph)
# class Graph:
#     def __init__(self, vertices):
#         self.V = vertices
#         self.edges = []

#     def add_edge(self, src, dest, weight):
#         self.edges.append((src, dest, weight))

#     def find_parent(self, parent, i):
#         if parent[i] == i:
#             return i
#         return self.find_parent(parent, parent[i])

#     def union(self, parent, rank, x, y):
#         x_root = self.find_parent(parent, x)
#         y_root = self.find_parent(parent, y)

#         if rank[x_root] < rank[y_root]:
#             parent[x_root] = y_root
#         elif rank[x_root] > rank[y_root]:
#             parent[y_root] = x_root
#         else:
#             parent[y_root] = x_root
#             rank[x_root] += 1

#     def kruskal_mst(self):
#         result = []
#         i = 0  # Sıralanmış kenarları takip etmek için indeks
#         e = 0  # MST'ye eklenen kenar sayısını takip etmek için sayaç

#         self.edges = sorted(self.edges, key=lambda x: x[2])  # Kenarları ağırlıklarına göre sırala

#         parent = []
#         rank = []

#         for node in range(self.V):
#             parent.append(node)
#             rank.append(0)

#         while e < self.V - 1:
#             src, dest, weight = self.edges[i]
#             i += 1

#             x = self.find_parent(parent, src)
#             y = self.find_parent(parent, dest)

#             if x != y:
#                 e += 1
#                 result.append((src, dest, weight))
#                 self.union(parent, rank, x, y)

#         return result


# # bkz
# g = Graph(4)
# g.add_edge(0, 1, 10)
# g.add_edge(0, 2, 6)
# g.add_edge(0, 3, 5)
# g.add_edge(1, 3, 15)
# g.add_edge(2, 3, 4)

# mst = g.kruskal_mst()
# for edge in mst:
#     print(f"{edge[0]} -- {edge[1]} == {edge[2]}")

#Sliding Window
# def find_max_sum_subarray(arr, k):
#     window_start = 0
#     window_sum = 0
#     max_sum = float('-inf')

#     for window_end in range(len(arr)):
#         window_sum += arr[window_end]

#         if window_end >= k - 1:
#             max_sum = max(max_sum, window_sum)
#             window_sum -= arr[window_start]
#             window_start += 1

#     return max_sum


# # bkz
# arr = [2, 1, 5, 1, 3, 2]
# k = 3

# max_sum = find_max_sum_subarray(arr, k)
# print("Alt dizi içindeki maksimum toplam:", max_sum)

