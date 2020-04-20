import tkinter,tkinter.messagebox
import os
import time
import zipfile
from PIL import Image, ImageTk

def backup():
    print("start backing up")
    # 获取压缩地址
    global entry_source
    global entry_target
    sourcePath = entry_source.get()
    targetPath = entry_target.get()
    print(sourcePath," is backed up to ", targetPath)

    path_confirm = tkinter.messagebox.askyesno("Hint","File in {} -> {}".format(sourcePath, targetPath))

    # 确认是否压缩
    if path_confirm == True:

        if not os.path.exists(targetPath):
            os.mkdir(targetPath)
        today = targetPath + os.sep + time.strftime('%Y%m%d')       # os.sep 表示win \ , linux /
        now = time.strftime('%H%M%S')
        target = today + os.sep + now + '.zip'

        print("文件：：", [today, now, target])

        if not os.path.exists(today):
            os.mkdir(today)
            print('success to create directory ', today)

        tarZip = zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED)

        # 提取需要打包的文件
        fileList, root0, dir0 = [], [], []
        for root, dirs, files in os.walk(sourcePath):
            # os.walk: top -- 根目录下的每一个文件夹(包含它自己), 产生3-元组 (dirpath, dirnames, filenames)【文件夹路径, 文件夹名字, 文件名】
            root0.append(root)
            dir0.append(dirs)
            for file in files:
                fileList.append(os.path.join(root, file))


        print("文件夹路径：", root0)
        # print("文件夹名字：", dir0)
        print("文件名： ", fileList)  # 路径下的拷贝文件

        for filename in fileList:
            tarZip.write(filename, filename[len(sourcePath):])
        tarZip.close()
        print('compress file successfully!')

    else:
        print("Cancel File in {} will zip to {}".format(sourcePath, targetPath))


root = tkinter.Tk()
root.title('BackUp')
root.geometry("250x150")
#第一行的两个控件
lbl_source = tkinter.Label(root, text='Source')
lbl_source.grid(row=0, column=0)
entry_source = tkinter.Entry(root)
entry_source.place(width=300, height=20)
entry_source.grid(row=0,column=1)
#第二行的两个控件
lbl_target = tkinter.Label(root, text='Target')
lbl_target.grid(row=1, column=0)
entry_target = tkinter.Entry(root)
entry_target.grid(row=1,column=1)
#第三行的一个按钮控件
but_back = tkinter.Button(root,text='BackUp')
but_back.grid(row=1, column=3)
but_back["command"] = backup
#界面的开始

im = Image.open(r"D:\git-project\personal project\project list\back_documents\source\back.jpg")
im = im.resize((100, 100))
img = ImageTk.PhotoImage(im)
imLabel = tkinter.Label(root, image=img).grid(row=2, column=1)  # 全局变量

root.mainloop()