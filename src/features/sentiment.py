# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import time

from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = '9812995'
API_KEY = 'nU5tVtSmk1lwipR1E3bDD16F'
SECRET_KEY = '8Pji3fROt7XBGhddkeX9Bvmi37yHdnKR'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)


def sentiment_classify(text):
    """
    调用情感倾向分析
    :param text:
    :return:
    """
    # result:
    # {
    #     "text":"苹果是一家伟大的公司",
    #     "items":[
    #         {
    #             "sentiment":2,    //表示情感极性分类结果, 0:负向，1:中性，2:正向
    #             "confidence":0.40, //表示分类的置信度
    #             "positive_prob":0.73, //表示属于积极类别的概率
    #             "negative_prob":0.27  //表示属于消极类别的概率
    #         }
    #     ]
    # }
    # time.sleep(0.5)
    result = {"sentiment": 1,
              "confidence": 0.0,
              "positive_prob": 0.0,
              "negative_prob": 0.0}
    # try:
    #     r = client.sentimentClassify(text)
    #     if r and 'items' in r:
    #         result = r['items']
    # except UnicodeEncodeError as e:
    #     print("error", e)
    # except Exception as es:
    #     print("error", es)
    return result


if __name__ == '__main__':
    texts = ["苹果是一家伟大的公司", "我刚去吃饭了", "你到底在干嘛呀", "我以前是个小偷，现在是个警察了",
             "再也不敢吃紫菜了",
             "我正在刘鑫描述凶案事发过程江歌母亲在日请愿判凶手死刑的主页，朋友们快来看看吧",
             "王者荣耀游戏风靡校园。据了解，在上海和国内不少城市的中小学校，玩王者荣耀的学生比例不低。有小学生称，"
             "班里有同学为了玩游戏，凌晨3点起床，一直打到6点，再去上学。专家表示，中小学生痴迷游戏并非好事，父母要适当引导和干预。"]
    for i in texts:
        r = sentiment_classify(i)
        print(i, r)
