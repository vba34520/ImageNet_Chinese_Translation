import json
import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing import image

model = mobilenet_v2.MobileNetV2(weights='imagenet')  # 加载预训练模型
initialdir = Path.cwd()  # 初始化目录，可切换为图片Path.home() / 'Pictures'
img = None  # 当前打开的图片
win_result = None  # 显示结果的窗口
class_trans_path = 'class_trans.json'  # 翻译文件路径
class_trans = json.load(open(class_trans_path)) if Path(class_trans_path).exists() else {}


def scale(size, width=None, height=None):
    """获取按比例缩放后的宽高"""
    if not width and not height:
        width, height = size
    if not width or not height:
        _width, _height = size
        height = width * _height / _width if width else height
        width = height * _width / _height if height else width
    return int(width), int(height)


def img_resize(event=None):
    """显示图片"""
    global img
    if img:
        _img = img.resize(scale(img.size, height=win.winfo_height()))
        _img = ImageTk.PhotoImage(_img)
        label.config(image=_img)
        label.image = _img


def close_win_result():
    """关闭结果窗口"""
    global win_result
    if win_result:
        try:
            win_result.destroy()
        except:
            pass


def on_closing():
    """关闭事件"""
    if messagebox.askokcancel('关闭', '是否退出程序？'):
        win.destroy()
        close_win_result()


def open_file():
    """打开图片"""
    global initialdir
    global img
    global win_result
    file_path = filedialog.askopenfilename(title='选择图片', initialdir=initialdir,
                                           filetypes=[('image files', ('.png', '.jpg', '.jpeg', '.gif'))])
    if file_path:
        path = Path(file_path)
        initialdir = path.parent
        img = Image.open(file_path)
        img_resize()

        _img = image.load_img(file_path, target_size=(224, 224))
        _img = image.img_to_array(_img)
        _img = np.expand_dims(_img, axis=0)
        _img = mobilenet_v2.preprocess_input(_img)
        pred_class = model.predict(_img)
        n = 10
        top_n = mobilenet_v2.decode_predictions(pred_class, top=n)
        print(path)
        for i in top_n[0]:
            print(i)
        print()

        close_win_result()
        win_result = tk.Tk()
        win_result.title(path.name)
        table = ttk.Treeview(win_result, columns=['序号', '对象', '标签', '翻译', '概率'], show='headings')
        table.column('序号', width=100)
        table.column('对象', width=100)
        table.column('标签', width=100)
        table.column('翻译', width=100)
        table.column('概率', width=100)
        table.heading('序号', text='序号')
        table.heading('对象', text='对象')
        table.heading('标签', text='标签')
        table.heading('翻译', text='翻译')
        table.heading('概率', text='概率')
        for i, x in enumerate(top_n[0]):
            index = str(i + 1)
            objectname = x[0]
            classname = x[1]
            transname = class_trans.get(classname, classname)
            table.insert('', i, text=index,
                         values=[index, objectname, classname, transname, '{:.2f}%'.format(float(x[2] * 100))])
        table.pack(fill=tk.BOTH, expand=True)
        win_result.mainloop()


win = tk.Tk()
win.title('ImageNet图像分类')  # 标题
menu = tk.Menu(win)
menu.add_command(label='打开', command=open_file)
win.config(menu=menu)
label = tk.Label(win, text='左上角打开图片')
label.pack(fill=tk.BOTH, expand=True)
win.bind('<Configure>', img_resize)
win.geometry('600x300+300+300')
win.minsize(200, 200)
win.protocol('WM_DELETE_WINDOW', on_closing)
win.mainloop()
