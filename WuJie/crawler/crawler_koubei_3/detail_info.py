import re
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fontTools.ttLib import TTFont
import pandas as pd
import xlwt
from fake_useragent import UserAgent
import time
import random
import openpyxl

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

################################这一部分用来获取完整的文本信息，是parse函数中的子函数###################################
def comp(l1,l2):  #定义一个比较函数，比较两个列表的坐标信息是否相同
    if len(l1)!=len(l2):
        return False
    else:
        mark=1
        for a in range(len(l1)):
            if abs(l1[a][0]-l2[a][0])<40 and abs(l1[a][1]-l2[a][1])<40:
                pass
            else:
                mark=0
                break
        return mark


def font_list():
    u_list=['uniEC6D','uniECBF','uniEDFF','uniED4C','uniED9E','uniECEA','uniED3C','uniEC89','uniEDC9','uniEC27','uniED68','uniECB4','uniED06','uniEC53',
            'uniECA5','uniEDE5','uniED32','uniED84','uniECD0','uniEC1D','uniEC6F','uniEDAF','uniEE01','uniED4E','uniEC9A','uniECEC','uniEC39','uniED79',
            'uniEDCB','uniED18','uniED6A','uniECB6','uniEDF7','uniEC55','uniED95','uniECE2','uniED34','uniEC80','uniECD2','uniEC1F','uniED5F','uniEDB1',
            'uniECFE','uniEC4A','uniEC9C','uniEDDD','uniEC3A','uniED7B','uniECC8','uniED19','uniEC66','uniEDA7','uniEDF8','uniED45','uniED97','uniECE3',
            'uniEC30','uniEC82','uniEDC2','uniED0F','uniED61','uniECAD','uniECFF','uniEC4C','uniED8C','uniEDDE','uniED2B','uniEC77','uniECC9','uniEE0A',
            'uniEC68','uniEDA8','uniECF5','uniED47','uniEC93','uniEDD4','uniEC32','uniED72','uniEDC4','uniED11','uniEC5D','uniECAF','uniEDF0','uniED8E',
            'uniECDB','uniED2C','uniEC79','uniEDBA','uniEE0B','uniED58'
            ]
    word_list=['排','高','保','灯','九','自','有','一',
               '泥','四','真','更','比','启','当','里',
               '问','光','开','无','空','只','档','是',
               '地','少','和','短','远','下','控','味',
               '外','过','大','级','加','盘','七','耗',
               '公','硬','五','响','内','小','长','十',
               '坏','养','身','三','好','油','上','副',
               '路','着','不','机','手','来','性','的',
               '中','冷','软','低','八','坐','近','呢',
               '了','多','很','门','音','孩','得','二',
               '六','量','实','右','左','动','皮','电',
               '雨','矮'
               ]

    font1=TTFont('0.ttf')
    be_p1=[]  #保存38个字符的（x,y）信息
    for uni in u_list:
        p1 = []
        p=font1['glyf'][uni].coordinates
        for f in p:
            p1.append(f)
        be_p1.append(p1)

    font2=TTFont('1.ttf')
    uni_list2=font2.getGlyphOrder()[1:]
    on_p1=[]
    for uni_list in uni_list2:
        pp1 = []
        p=font2['glyf'][uni_list].coordinates
        for f in p:
            pp1.append(f)
        on_p1.append(pp1)

    n2=0
    x_list=[]
    for d in on_p1:
        n2+=1
        n1=0
        for a in be_p1:
            n1+=1
            if comp(a,d):
                x_list.append(word_list[n1-1])
    return x_list

def ALL_Comment_Url(excelpath):
    id_list=[]
    url_list=[]
    df = pd.DataFrame(pd.read_excel(excelpath))
    for id in df["编号"]:
        id_list.append(id)
    for url in df["链接"]:
        url_list.append(url)
    return url_list, id_list


# 抓取文本内容
class AutoSpider:
    # 页面初始化
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('Accept="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"')
        self.chrome_options.add_argument('Accept-Encoding="gzip, deflate, br"')
        self.chrome_options.add_argument('Accept-Language="zh-CN,zh;q=0.9,en;q=0.8"')
        self.chrome_options.add_argument('Cache-Control="max-age=0"')
        self.chrome_options.add_argument('Connection="keep-alive"')
        self.chrome_options.add_argument('Upgrade-Insecure-Requests="1"')
        self.chrome_options.add_argument('User-Agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"')
        # self.chrome_options.add_argument('User-Agent={str(UserAgent(use_cache_server=False).random)}')
        self.chrome_options.add_argument('Host="k.autohome.com.cn"')
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        prefs = {'profile.managed_default_content_settings.images': 2}
        self.chrome_options.add_experimental_option('prefs', prefs)
        # self.driver = webdriver.Chrome(options=self.chrome_options, executable_path='C:/Users/dell/AppData/Local/Programs/Python/Python37/Lib/site-packages/selenium/webdriver/chrome/chromedriver.exe')

    # 获取文本
    def getNote(self,url):
        random_sleep()
        self.driver = webdriver.Chrome(options=self.chrome_options, executable_path='C:/Softwares/coding/python/Lib/site-packages/selenium/webdriver/chrome/chromedriver.exe')
        self.driver.get(url)

        # 获取页面内容
        text = self.driver.find_element_by_tag_name('html').get_attribute('outerHTML')

        # 匹配ttf font
        cmp = re.compile("url\('(//.*.ttf)'\) format\('woff'\)")
        rst = cmp.findall(text)
        ttf = requests.get("http:" + rst[0], stream=True)
        with open("1.ttf", "wb") as pdf:
            for chunk in ttf.iter_content(chunk_size=1024):
                if chunk:
                    pdf.write(chunk)
        # 解析字体库font文件
        font = TTFont('1.ttf')
        uniList = font['cmap'].tables[0].ttFont.getGlyphOrder()
        utf8List = [eval("'\\u" + uni[3:] + "'").encode("utf-8") for uni in uniList[1:]]
        wordList = font_list()

        # 获取文本内容
        note_list = self.driver.find_elements_by_class_name('text-con')
        # print("note_list = ", note_list)
        note = ''
        for i in note_list:
            note = note+i.text+'\n'
        for i in range(len(utf8List)):
            note = note.encode("utf-8").replace(utf8List[i], wordList[i].encode("utf-8")).decode("utf-8")
        # print("note = ", note)
        note = ILLEGAL_CHARACTERS_RE.sub(r'', note)
        print(note != '')
        return note

    def driver_close(self):
        self.driver.quit()


def write_to_excel(id_index, comment, car, counter):
    # work=xlwt.Workbook()
    work = openpyxl.Workbook()
    # sheet=work.add_sheet('口碑帖子',cell_overwrite_ok=True)
    sheet = work.active
    sheet.append(["编号", "帖子"])
    # sheet.write(0,0,"编号")
    # sheet.write(0,1,"帖子")
    for i in range(len(id_index)):
        # print("i = ", i)
        # print("id_index[",i,"] = ", id_index[i])
        # sheet.write(i+1,0,id_index[i])
        # sheet.write(i+1,1,comment[i])
        sheet.append([id_index[i], comment[i]])
    work.save('./DATA/' + str(car) + "_" + str(counter) + '.xlsx')  # 根据自己的路径修改把excel保存到特定的路径下


def random_sleep(mu=3, sigma=1):
    '''正态分布随机睡眠
    :param mu: 平均值
    :param sigma: 标准差，决定波动范围
    '''
    secs = random.normalvariate(mu, sigma)
    if secs <= 0:
        secs = mu  # 太小则重置为平均值
    time.sleep(secs)


def main():
    dtl_spider = AutoSpider()
    carlist = [2]  # 'E1','E3','E5','E6','han','qin','qinpro','song','songmax','songpro',
    for car in carlist:
        id_index = []
        comment = []
        print('#############################', car, '################################')
        excelpath = './URL/car_link'+str(car)+'.xlsx'
        all_url_list, all_id_list = ALL_Comment_Url(excelpath)
        index = 0
        counter = all_id_list[0]  # 记录已爬取数量，每200条写入一次文件
        for url in all_url_list:
            index += 1
            print(index)
            try:
                comment.append(dtl_spider.getNote(url))
            except:
                print('...'+str(404)+'...')
                random_sleep()
                continue
            id_index.append(all_id_list[index-1])
            counter += 1
            random_sleep()
            # 写入文件
            if counter % 200 == 0:
                write_to_excel(id_index, comment, car, counter)
                id_index = []
                comment = []
                print("counter =", counter, ",写入文件...")
        # write_to_excel(id_index, comment, car)
    dtl_spider.driver_close()


if __name__ == '__main__':
    main()
    print('爬取结束')

