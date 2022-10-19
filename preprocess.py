import argparse
import text
from utils import load_filepaths_and_text

if __name__ == '__main__':
    """
    用于预处理文本的函数，在本项目中没有被使用
    """
    parser = argparse.ArgumentParser()
    # 输出的文件拓展名
    parser.add_argument("--out_extension", default="cleaned")
    # 指定文本所在的数据顺序
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt",
                                                           "filelists/ljs_audio_text_test_filelist.txt"])
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    # 遍历参数中设置的用于训练的配置文件txt
    for filelist in args.filelists:
        print("START:", filelist)
        # 将配置文件加载为数组数据结构方便访问
        filepaths_and_text = load_filepaths_and_text(filelist)
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)
            filepaths_and_text[i][args.text_index] = cleaned_text

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
