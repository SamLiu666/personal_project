import os
import time
import tkinter
import zipfile


# # 操作可视化
# def window():
#     root = tkinter.Tk()
#     root.title('BackUp')
#     root.geometry("200x200")
#     # 第一行的两个控件
#     lbl_source = tkinter.Label(root, text='Source')
#     lbl_source.grid(row=0, column=0)
#     entry_source = tkinter.Entry(root)
#     entry_source.grid(row=0, column=1)
#     # 第二行的两个控件
#     lbl_target = tkinter.Label(root, text='Target')
#     lbl_target.grid(row=1, column=0)
#     entry_target = tkinter.Entry(root)
#     entry_target.grid(row=1, column=1)
#     # 第三行的一个按钮控件
#     but_back = tkinter.Button(root, text='BackUp')
#     but_back.grid(row=3, column=0)
#     but_back["command"] = compressZip()
#     # 界面的开始
#     root.mainloop()

# 文件压缩
def compressZip(sourcePath, targetPath):
    '''
    :param sourcePath:待压缩文件所在文件目录
    :param targetPath:目标文件目录
    :return:null
    '''
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
    print("文件夹名字：", dir0)
    print("文件名： ", fileList)  # 路径下的拷贝文件

    for filename in fileList:
        tarZip.write(filename, filename[len(sourcePath):])
    tarZip.close()
    print('compress file successfully!')


if __name__ == '__main__':
    # window()
    # 路径最好为绝对路径，方便查找
    souce = r'D:\git-project\test\1 web server\back_documents'
    target = r'D:\git-project\test\1 web server\save'
    compressZip(souce, target)