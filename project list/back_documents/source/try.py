import zipfile


def unzip(sourceFile, targetPath):
    '''
    :param sourceFile: 待解压zip路径
    :param targetPath: 目标文件目录
    :return:
    '''
    file = zipfile.ZipFile(sourceFile, 'r')
    file.extractall(targetPath)
    print('success to unzip file!')