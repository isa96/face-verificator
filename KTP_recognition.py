# load libraries
import os
import ocrspace

# define API
api = ocrspace.API(endpoint='https://api.ocr.space/parse/image',
                   api_key='c1876db79188957',
                   language=ocrspace.Language.English)


# input from image filename and result detection
def text_ocr(filename):
    return api.ocr_file(filename)


if __name__ == "__main__":
    folder ='utils/data_2/'
    for idx, file in enumerate(os.listdir(folder)):
        result = text_ocr(folder+file)
        print(idx)
        print(result)
