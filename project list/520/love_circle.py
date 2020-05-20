from PIL import Image, ImageDraw, ImageFont
def print_love():
    print('\n'.join([line for line in [''.join([('Love'[(x-y) % len('Love')] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0 else ' ') for x in range(-30, 30)]) for y in range(30, -30, -1)] if line != '                                                            ']))

def picture_with_text():
    """中文文本显示填充图片"""
    font_size, text = 9, "我爱你"
    input_img_path = "1.jpeg"  # 输入照片路径
    output_img_path = "love.png"  # 保存照片路径

    img_raw = Image.open(input_img_path)
    img_array = img_raw.load()

    img_new = Image.new("RGB", img_raw.size, (0, 0, 0))
    draw = ImageDraw.Draw(img_new)
    font_path='simsun.ttc'
    font = ImageFont.truetype(font_path, font_size)

    def character_generator(text):
        while True:
            for i in range(len(text)):
                yield text[i]

    ch_gen = character_generator(text)

    for y in range(0, img_raw.size[1], font_size):
        for x in range(0, img_raw.size[0], font_size):
            draw.text((x, y), next(ch_gen), font=font, fill=img_array[x, y], direction=None)


    img_new.convert('RGB').save(output_img_path)


if __name__ == '__main__':
    print_love()