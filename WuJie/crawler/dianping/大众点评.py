import re
import time

import requests
import random
from parsel import Selector
import pandas as pd


class DaZongDianPing:

    def __init__(self, url, Cookie, sec=2,name='',cycle=100):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.%s' % random.randint(
                1, 1000000),
            'Upgrade-Insecure-Requests':'1',
            'Host': 'www.dianping.com',
            'Connection':'keep-alive',
            "Cookie": Cookie,
            'Accept-Encoding': 'gzip;deflate',
        }
        self.url = url
        self.name = name
        self.cycle = cycle
        self.sec = sec
        self.data = []

    def main(self):
        try:
            print('开始爬取{}了.....'.format(self.name))
            for i in range(self.cycle):
                html = self.get_index(i + 1)
                css_url, class_name = self.get_url_and_tag(html)
                di = self.get_css_and_svg(css_url, class_name)
                num = self.parse_index(html, di)
                print('爬到第{}页 共{}个评论'.format(i + 1, num))
                time.sleep(self.sec)
            print('爬取{}结束了 保存中.....'.format(self.name))
            self.save_file()
        except Exception as ex:
            print(ex)
            print('爬取{}结束了 保存中.....'.format(self.name))
            self.save_file()

    def get_index(self, num):
        if num == 1:
            url = self.url + '/review_all'
        else:
            url = self.url + '/review_all/p' + str(num)
        print(url)
        resp = requests.get(url, headers=self.headers)
        if resp.status_code == 200:
            return resp.text

    def get_url_and_tag(self, html):
        '''获取css_url和网页中的加密字体标签的class名'''
        css_url = re.findall(r'href="(.*?svgtextcss.*?)"', html)
        if css_url:
            css_url = 'http:' + css_url[0]
        # 加密字体的class名
        class_name = re.findall(r'<svgmtsi class="(.*?)">', html)
        return css_url, class_name

    def get_css_and_svg(self, css_url, class_name):
        '''
        获取css属性和svg地址，根据css属性查找真实数据，构建替换字典
        svg地址有3个
        cc[class^="wgx"]  电话
        bb[class^="wnu"]  地址
        svgmtsi[class^="kvg"]  评论
        '''
        # http://s3plus.meituan.net/v1/mss_0a06a471f9514fc79c981b5466f56b91/svgtextcss/0ddbef571464748f0408df2fb2ac1756.css
        css_resp = requests.get(css_url).text.replace("\n", "").replace(" ", "")
        # 获取评论的svg地址
        svg_url = re.findall(r'svgmtsi.*?url\((.*?)\);', css_resp)
        if svg_url:
            svg_url = 'http:' + svg_url[0]
            svg_resp = requests.get(svg_url).text
        # 获取css属性值  对应的坐标值
        d = {}
        for name in class_name:
            # print('class_name', name)
            coord = re.findall(r"%s{background:-(.*?)px-(.*?)px;}" % name, css_resp)
            # print("coord", coord)
            x, y = coord[0]
            css_x, css_y = int(float(x)), int(float(y))
            # 获取svg标签对应的y值，规则是svg_y>=css_y
            svg_data = Selector(svg_resp)
            tests = svg_data.xpath('//text')
            # 3.如何选择svg_y？比较y坐标，选择大于等于css_y的最接近的svg_y
            svg_y = [i.attrib.get('y') for i in tests if css_y <= int(i.attrib.get('y'))][0]
            # 根据svg_y确定具体的text的标签
            svg_text = svg_data.xpath(f'//text[@y="{svg_y}"]/text()').extract_first()
            # 4、确认SVG中的文字大小
            font_size = re.findall(r'font-size:(\d+)px', svg_resp)[0]
            # 5、得到css样式vhkbvu属性映射svg的位置
            # css_x // 字体大小 的值就是数值的下标
            position = css_x // int(font_size)
            s = svg_text[position]
            d[name] = s
        # 加密字体整个标签与真实值之间的字典
        di = {f'<svgmtsi class="{k}"></svgmtsi>': v for k, v in d.items()}
        return di

    def parse_index(self, html, di):
        '''解析网页数据'''
        for key, value in di.items():
            if key in html:
                html = html.replace(key, value)
                # print(html)
        selector = Selector(html)
        # 评论摘要
        desc_li = selector.xpath('//div[@class="review-words Hide"]/text()').extract()
        num = 0
        for desc in desc_li:
            desc = desc.replace('\t', '').replace('\n', '').replace(' ', '')
            num += 1
            self.data.append(desc)
        return num

    # 保存爬取数据
    def save_file(self):
        path_file_name = str('{}的评论{}条.csv'.format(self.name,len(self.data)))
        file = pd.DataFrame(self.data, columns=['评论'])
        file.to_csv(path_file_name, index=False)


if __name__ == '__main__':
    # 爬取一页后休息多少秒
    sec = 3
    # 爬取商品名称
    name = '桐庐生仙里国际滑雪场'
    # 爬取商品页数
    cycle=134
    # 大众点评的商品页面
    url = 'http://www.dianping.com/shop/k9lZLOp5q6Cpqo93'
    # 大众点评需要手动登录 按F12 随便点击一个连接 点击网络 点击文档 点击标头 找到 右键复制到下方这样就算登录了
    Cookie = 'fspop=test; _lxsdk_cuid=17ef1322b4fc8-04783f93afbfa2-f791539-240000-17ef1322b4fc8; _lxsdk=17ef1322b4fc8-04783f93afbfa2-f791539-240000-17ef1322b4fc8; _hc.v=b4e94d23-79ae-8633-31c0-53943017ef40.1644724104; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1644724104; s_ViewType=10; _dp.ac.v=2767aaa0-20e9-456b-bcf7-5d7138a8501f; thirdtoken=6e9984a8-9ee4-44ee-a255-ea5d6760aadc; _thirdu.c=3ae227a9e677004a05f47ca09cd7fc2e; ctu=ef9f68d1228602ae124ba009360b49e090ef5fd9cc3780ed268582fdfe7e54f9; cy=3; cye=hangzhou; _lx_utm=utm_source=Baidu&utm_medium=organic; ua=dpuser_0593792719; uamo=18122933855; dplet=1a997739eb565edf24f59917a1e2e067; dper=b839ef162e44d13f94070d4e4059e16b6e10bdef37e64dd35d301a9bdc566b54b3cb0adad66dfc5a1a3bd0b0ad4ceb15de5342c8f8b9809fbe49e5e603c02238998935057258772186a111e3d236aff44d5f75d3ca0d59a3f3821224ef4ea9c6; ll=7fd06e815b796be3df069dec7836c3df; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1644741362; _lxsdk_s=17ef20daf42-3e3-008-66d||1419'
    d = DaZongDianPing(url, Cookie, sec,name,cycle)
    d.main()
