from MyQR import myqr


# 只能生成英文字符
def static_qrcode():
    myqr.run(
        words = "https://github.com/SamLiu666",     # 想要生成的内容
        picture="git.jpg",                          # 二维码加载图片
        colorized=True,
        save_name="test.png"                        # 存储名称
    )


myqr.run(
    words="https://github.com/SamLiu666",  # 想要生成的内容
    picture="wm.gif",  # 二维码加载图片
    colorized=True,
    save_name="test.gif"  # 存储名称
)