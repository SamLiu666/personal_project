from PIL import Image

im = Image.open("1.jpg")
# 转换为像素模式（8位）
im.convert("P")
im2 = Image.new("P", im.size, 255)
# 打印颜色图
# print(im.histogram())

his = im.histogram()
values = {}

for i in range(256):
    values[i] = his[i]

for i, j in sorted(values.items(), key=lambda x:x[1], reverse=True)[:10]:
    print(i,j)


for x in range(im.size[1]):
    for y in range(im.size[0]):
        pix = im.getpixel((y, x))
        if pix == 255 or pix == 254:
            im2.putpixel((y,x), 0)

im2.save("new11.gif")