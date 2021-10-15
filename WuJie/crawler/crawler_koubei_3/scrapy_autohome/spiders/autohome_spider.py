# -*- coding: utf-8 -*-
import scrapy
from scrapy_autohome.all_car_id import All_Car_Id
from scrapy_autohome.items import ScrapyAutohomeItem
import re

class AutohomeSpider(scrapy.Spider):
    name = "autohome_spider"
    allowed_domains = ["autohome.com"]
    start_urls = ['http://autohome.com/']
    # 评论的个数
    count = 0

    # 循环页码，就在这个函数中实现。
    def start_requests(self):
        reqs = []  # 每个车型页面的request

        # 获取所有车辆的ID
        all_car_id = All_Car_Id()
        car_id_list = all_car_id.car_id_list
        # 两层遍历，分别遍历车型和页数
        for i in car_id_list:  # i代表从车型的遍历
            print(i)
            for j in range(1,46): # j代表评论页数，range(1,3)表示1到2页
                req = scrapy.Request("http://k.autohome.com.cn/"+str(i)+"/StopSelling/index_"+str(j)+".html#dataList")
                reqs.append(req)
        return reqs

    def parse(self, response):
        # 记录个数
        # AutohomeSpider.count += 1
        #print(AutohomeSpider.count)

        # 获取所有评论div //*[@id="maodian"]/div/div/div[2]/div[4]
        divs = response.xpath('//*[@id="maodian"]/div/div/div[2]/div[@class="mouthcon"]')


        for div in divs:
            # 记录个数
            AutohomeSpider.count += 1
            print("----------------------------------")
            print("第：",AutohomeSpider.count,"个评论。")

            item = ScrapyAutohomeItem()
            # 车ID //*[@id="maodian"]/div/div/div[2]/div[4]/div/div[1]/div[2]/dl[1]/dd/a[1]

            item['CAR_ID'] = div.xpath('div/div[1]/div[2]/dl[1]/dd/a[1]/@href')[0].extract().replace('/','')
            print('车型ID',item['CAR_ID'])
            # print('车ID',item['CAR_ID'])
            # 车名字
            item['CAR_NAME'] = div.xpath('div/div[1]/div[2]/dl[1]/dd/a[1]/text()')[0].extract()
            # print('车名字',item['CAR_NAME'])
            # 用户ID  //*[@id="maodian"]/div/div/div[2]/div[4]/div/div[1]/div[1]/div/div[1]/div[2]/p/a
            USER_ID1 = div.xpath('div/div[1]/div[1]/div/div[1]/div[2]/p/a/@href')[0].extract()
            item['USER_ID'] = re.findall('\d{1,15}',USER_ID1)[0]
            item['USER_NAME'] = div.xpath('div/div[1]/div[1]/div/div[1]/div[2]/p/a/text()')[0].extract().strip()
            print('用户名',item['USER_NAME'])

            # 购买地点 //*[@id="maodian"]/div/div/div[2]/div[4]/   div/div[1]/div[2]/dl[2]/dd
            PURCHASE_PLACE = div.xpath('div/div[1]/div[2]/dl[2]/dd')[0]
            item['PURCHASE_PLACE'] =PURCHASE_PLACE.xpath('string(.)').extract()[0].strip()
            # print('购买地点',item['PURCHASE_PLACE'])

            # 因为列表属性相同且数量不确定，所要加入判断
            dls =div.xpath('div/div[1]/div[2]/dl')
            # 正常的有7个
            if dls.__len__() == 7:

                # 购买时间 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[4]/dd
                item['PURCHASE_TIME'] = div.xpath('div/div[1]/div[2]/dl[4]/dd/text()')[0].extract().strip()

                # 裸车购买价 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[5]/dd
                CAR_PRICE = div.xpath('div/div[1]/div[2]/dl[5]/dd')[0]
                item['CAR_PRICE'] = CAR_PRICE.xpath('string(.)').extract()[0].strip().replace('\xa0','')

                # 购车目的 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[7]/dd
                PURCHASE_PURPOSE = div.xpath('div/div[1]/div[2]/dl[7]/dd')[0]
                item['PURCHASE_PURPOSE'] = PURCHASE_PURPOSE.xpath('string(.)').extract()[0].strip().replace('\r\n','').replace('                                ',';')

            #不正常的有6个，分为两种情况：缺经销商和缺油耗。
            elif dls.__len__() == 6:

                p = div.xpath('div/div[1]/div[2]/dl[5]/dt/p')

                # 如果有p标签 ，说明有油耗，没有经销商
                if p:
                    # 购买时间 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[4]/dd
                    item['PURCHASE_TIME'] = div.xpath('div/div[1]/div[2]/dl[3]/dd/text()')[0].extract().strip()

                    # 裸车购买价 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[5]/dd
                    CAR_PRICE = div.xpath('div/div[1]/div[2]/dl[4]/dd')[0]
                    item['CAR_PRICE'] = CAR_PRICE.xpath('string(.)').extract()[0].strip().replace('\xa0', '')

                    # 购车目的 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[7]/dd
                    PURCHASE_PURPOSE = div.xpath('div/div[1]/div[2]/dl[6]/dd')[0]
                    item['PURCHASE_PURPOSE'] = PURCHASE_PURPOSE.xpath('string(.)').extract()[0].strip().replace('\r\n','').replace('                                ', ';')

                # 如果没有p说明 没有油耗，有经销商
                else:
                    # 购买时间 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[4]/dd
                    item['PURCHASE_TIME'] = div.xpath('div/div[1]/div[2]/dl[4]/dd/text()')[0].extract().strip()

                    # 裸车购买价 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[5]/dd
                    CAR_PRICE = div.xpath('div/div[1]/div[2]/dl[5]/dd')[0]
                    item['CAR_PRICE'] = CAR_PRICE.xpath('string(.)').extract()[0].strip().replace('\xa0', '')

                    # 购车目的 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/dl[7]/dd
                    PURCHASE_PURPOSE = div.xpath('div/div[1]/div[2]/dl[6]/dd')[0]
                    item['PURCHASE_PURPOSE'] = PURCHASE_PURPOSE.xpath('string(.)').extract()[0].strip().replace('\r\n','').replace('                                ', ';')


            # print('购买目的',item['PURCHASE_PURPOSE'])
            # 评分- 空间 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/div[1]/dl/dd/span[2]
            item['SCORE_SPACE'] = div.xpath('div/div[1]/div[2]/div[1]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 动力 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/div[2]/dl/dd/span[2]
            item['SCORE_POWER'] = div.xpath('div/div[1]/div[2]/div[2]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 操控
            item['SCORE_CONTROL'] = div.xpath('div/div[1]/div[2]/div[3]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 油耗
            item['SCORE_FUEL_CONSUMPTION'] = div.xpath('div/div[1]/div[2]/div[4]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 舒适性
            item['SCORE_COMFORT'] = div.xpath('div/div[1]/div[2]/div[5]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 外观
            item['SCORE_EXTERIOR'] = div.xpath('div/div[1]/div[2]/div[6]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 内饰
            item['SCORE_INTERIOR'] = div.xpath('div/div[1]/div[2]/div[7]/dl/dd/span[2]/text()')[0].extract()

            # 评分- 性价比 //*[@id="maodian"]/div/div/div[2]/div[4]/ div/div[1]/div[2]/div[8]/dl/dd/span[2]
            item['SCORE_COST_EFFECTIVE'] = div.xpath('div/div[1]/div[2]/div[8]/dl/dd/span[2]/text()')[0].extract()
            # print('性价比',item['SCORE_COST_EFFECTIVE'])


            # 有多少人支持这条口碑  #//*[@id="maodian"]/div/div/div[2]/div[6]/  div/div[2]/div[1]/div[3]/div[2]/span[3]/label
            item['COMMENT_SUPPORT_QUANTITY'] = div.xpath('div/div[2]/div[1]/div[3]/div[2]/span[1]/label/text()')[0].extract()
            # print('支持数',item['COMMENT_SUPPORT_QUANTITY'])
            # 有多少人看过这条口碑  #//*[@id="maodian"]/div/div/div[2]/div[6]/  div/div[2]/div[1]/div[3]/div[2]/span[4]/a
            item['COMMENT_SEEN_QUANTITY'] = div.xpath('div/div[2]/div[1]/div[3]/div[2]/span[2]/a/text()')[0].extract()
            print('观看数',item['COMMENT_SEEN_QUANTITY'])

            url_post= div.xpath('div/div[2]/div[1]/div[2]/div[1]/div/a/@href')[0].extract()
            item['COMMENT_URL'] = "http:"+url_post
            yield item