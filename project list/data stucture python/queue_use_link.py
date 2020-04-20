class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return self.head is None

    def enqueue(self, data):
        node = Node(data)
        if self.isEmpty():
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def dequeue(self):
        if self.isEmpty():
            print("Empty Queue")
        else:
            out = self.head.data
            self.head = self.head.next
            return out

    def peek(self):
        if self.isEmpty():
            print("None")
        else:
            return self.head.data

    def print_queue(self):
        temp, res = self.head, []
        while temp:
            res.append(temp.data)
            temp = temp.next
        print(res)


class Queue_array:
    def __init__(self):
        self.queue = []
        self.length = 0
        self.front = 0

    def enqueue(self,data):
        self.queue.append(data)
        self.length += 1

    def dequeue(self):
        self.length -= 1
        dequeued = self.queue[self.front]
        self.front -= 1
        self.res = self.res[self.front:]
        return dequeued

    def peek(self):
        return self.res[0]

if __name__ == "__main__":
    queue = Queue()
    queue.enqueue(21)
    queue.enqueue(35)
    queue.enqueue(58)
    queue.enqueue(13)
    queue.print_queue()
    print(queue.peek())
    queue.dequeue()
    queue.print_queue()
    print(queue.peek())