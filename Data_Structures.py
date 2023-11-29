#Data Structures
#List -değiştirilebilir (mutable)
urunler = ['banana', 'cherry', 'apple', 'orange', 'grape', 'pineapple', 'pear', 5]
urunler.append(7)
del(urunler[0])
print(urunler)
print(type(urunler))


#Tuple -değiştirilemez (immutable)
servisler = (5, 'internet', 'telefon', 'masaüstü', 'bilgisayar')
print(servisler)
print(type(servisler))


#String
menu = "pizza"
print(type(menu))
print(len(menu))


#Sets
insanlar = {'ali', 'ayse', 'ali'}
insanlar.add('murat')
insanlar.pop()
print(insanlar)


#Dict
calisan = {"name": "John", "age": 30, "city": "New York"} 
calisan["name"] = "aly"
print(calisan)
print(calisan["name"])


#Stacks -LIFO
stack = ["a", "b", "c"]
stack.append("d")
stack.append("e")
stack.pop()
print(stack)


#Queues -FIFO
queue = ["a", "b", "c"]
queue.append("d")
queue.append("e")
queue.pop(0)
print(queue)


#LinkedList
from collections import deque
linked_list = deque() 
linked_list.append(1)
linked_list.append(2)
linked_list.append(3)

print(linked_list)











