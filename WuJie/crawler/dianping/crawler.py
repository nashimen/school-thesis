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
    sec = 15
    # 爬取商品名称
    name = '桐庐生仙里国际滑雪场'
    # 爬取商品页数
    cycle=134
    # 大众点评的商品页面
    url = 'http://www.dianping.com/shop/k9lZLOp5q6Cpqo93'
    # 大众点评需要手动登录 按F12 随便点击一个连接 点击网络 点击文档 点击标头 找到 右键复制到下方这样就算登录了
    Cookie = '_lxsdk_cuid=1805e6e82c9c8-050ff1b0a32b2e-6b3e555b-e1000-1805e6e82c9c8; _lxsdk=1805e6e82c9c8-050ff1b0a32b2e-6b3e555b-e1000-1805e6e82c9c8; _hc.v=46c40925-5a8c-8a78-3fc0-b75f171364a1.1650851743; ctu=ee2aa30807129017af0768a77ee119d2c5e8e7bc6fd14f5fb296260c8bb98301; s_ViewType=10; fspop=test; cy=2; cye=beijing; _lx_utm=utm_source=Baidu&utm_medium=organic; perf_dv5Tr4n=1; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1657715729,1657787858,1658123379,1658124430; WEBDFPID=395z5v7wx6535u0uzyx48vw0x05w32758179uvy722897958z85w7uv2-1658210846406-1658124444309UGAEMEKfd79fef3d01d5e9aadc18ccd4d0c95074095; dplet=c9cdb4538b187b11173820168a0deb09; dper=998740308e674632ec550f6caff3904be50d33bf5ad089bb504194f28cb4416a16bc589fc6e96bcdf620efdad77030bdf51a0ec7b04924a307c24d9d9f1cd5f26e2d13611062a6249240c276ace3f0b7d561ae7b27c7195e3f8b811af029cd43; ll=7fd06e815b796be3df069dec7836c3df; ua=武小杰; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1658125295; _lxsdk_s=1820fdadd22-207-569-9ab||1585'
    d = DaZongDianPing(url, Cookie, sec, name, cycle)
    d.main()
