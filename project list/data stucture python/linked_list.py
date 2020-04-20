class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Linked_list:

    def __int__(self, head=None):
        self.head = head

    def append(self, data):
        node = Node(data)
        current = self.head

        if self.head:
            while current.next:
                current = current.next
            current.next = node
        else:
            self.head = node

    def isEmpty(self):
        return not self.head

    def insert(self, position, data):
        node = Node(data)
        if position<0 or position>self.get_length():
            raise IndexError("Exceed Index")

        temp = self.head

        if position==0:
            node.next = temp
            self.head = node

        i = 0
        while i < position:
            pre = temp
            temp = temp.next
            i += 1
        pre.next = node
        node.next = temp

    def remove(self, position):
        if position < 0 or position > self.get_length() - 1:
            # print("insert error")
            raise IndexError('删除元素的索引超出范围')

        i = 0
        temp = self.head
        while temp:
            if position==0:
                self.head = temp.next
                temp.next = None
                return

            pre, temp = temp, temp.next
            i += 1
            if i==position:
                pre.next = temp.next
                temp.next = None

    def get_length(self):
        temp = self.head
        length = 0
        while temp:
            length += 1
            temp = temp.next
        return length

    def print_list(self):
        "打印链表"
        print("linked list: ")
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next

    def reverse(self):
        """翻转链表"""
        prev, current = None, self.head

        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

    def initlist(self, list_d):
        """ 将列表转换未链表"""
        self.head = Node(list_d[0])
        temp = self.head
        for i in list_d[1:]:
            node = Node(i)
            temp.next = node
            temp = temp.next


class Linkedlist:
    def __init__(self):
        self.head = None

    def print_list(self):  # 遍历链表，并将元素依次打印出来
        print("linked_list:")
        temp = self.head
        new_list = []
        while temp:
            new_list.append(temp.data)
            temp = temp.next
        print(new_list)

    def insert(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    def swapNodes(self, d1, d2):
        prevD1 = None
        prevD2 = None
        if d1 == d2:
            return
        else:
            D1 = self.head
            while D1 is not None and D1.data != d1:
                prevD1 = D1
                D1 = D1.next
            D2 = self.head
            while D2 is not None and D2.data != d2:
                prevD2 = D2
                D2 = D2.next
            if D1 is None and D2 is None:
                return
            if prevD1 is not None:
                prevD1.next = D2
            else:
                self.head = D2
            if prevD2 is not None:
                prevD2.next = D1
            else:
                self.head = D1
            temp = D1.next
            D1.next = D2.next
            D2.next = temp


if __name__ == '__main__':
    list = Linkedlist()
    list.insert(5)
    list.insert(4)
    list.insert(3)
    list.insert(2)
    list.insert(1)
    list.print_list()
    list.swapNodes(1, 4)
    print("After swapping")
    list.print_list()