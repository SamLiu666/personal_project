import threading, time


def h1():
    time.sleep(0.02)

def main1():
    for i in range(100):
        h1()

def main2():
    thread_list = []
    for i in range(100):
        t = threading.Thread(target=h1)
        t.start()
        thread_list.append(t)
    for j in thread_list:
        j.join()

if __name__ == '__main__':
    start = time.time()
    main1()
    end = time.time()
    print("单线程耗时：", end - start)

    start =time.time()
    main2()
    end = time.time()
    print("多线程耗时:  ", end-start)