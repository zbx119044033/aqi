# -*- coding:utf-8 -*-
# spider for pollute data from https://www.aqistudy.cn

import scrapy


class PolluteSpider(scrapy.Spider):
    name = "pollute"
    allowed_domains = ["aqistudy.cn"]
    start_urls = [u"https://www.aqistudy.cn/historydata/daydata.php?city=北京&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=承德&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=张家口&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=保定&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=廊坊&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=天津&month=201312",
                  u"https://www.aqistudy.cn/historydata/daydata.php?city=唐山&month=201312"]
    custom_settings = {
        'DOWNLOAD_DELAY': 6,
        'SPIDER_MIDDLEWARES': {
            'product.middlewares.spidermiddlewares.ProductSpiderMiddleware': 10,
        },
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            'product.middlewares.downloader.UAPool': 400,
            'product.middlewares.downloader.JavaScriptMiddleware': 390,
        },
        'ITEM_PIPELINES': {
            'product.pipelines.pipeline.FilterPipeline': 300,
            'product.pipelines.pipeline.JsonWriterPipeline': 800,
        }
    }

    def parse(self, response):
        next_pages = response.xpath("//ul[@class='unstyled1']/li/a/@href").extract()
        # print "next_pages=", next_pages
        for next_page_url in next_pages:
            next_page_url = response.urljoin(next_page_url)
            print("next_url=", next_page_url)
            request = scrapy.Request(next_page_url, callback=self.parse_item, dont_filter=True)
            request.meta['PhantomJS'] = True
            yield request

    def parse_item(self, response):
        data = []
        classes = {u'优': 1, u'良': 2, u'轻度污染': 3, u'中度污染': 4, u'重度污染': 5, u'严重污染': 6}
        i = 1
        for line in response.xpath("//tbody/tr"):
            if i == 1:
                i += 1
                continue
            else:
                date = int(line.xpath(".//td[1]/text()").extract_first().replace(u'-', u''))
                rank = classes[line.xpath(".//td[3]/span/text()").extract_first()]
                content = {u'date': date,  # int, 20131202
                           u'aqi': int(line.xpath(".//td[2]/text()").extract_first()),  # int
                           u'class': rank,  # 1, 2, 3, 4, 5, 6
                           u'pm2.5': int(line.xpath(".//td[4]/text()").extract_first()),  # int
                           u'pm10': int(line.xpath(".//td[5]/text()").extract_first()),  # int
                           u'so2': int(line.xpath(".//td[6]/text()").extract_first()),  # int
                           u'co': float(line.xpath(".//td[7]/text()").extract_first()),  # float
                           u'no2': int(line.xpath(".//td[8]/text()").extract_first()),  # int
                           u'o3': int(line.xpath(".//td[9]/text()").extract_first())}  # int
                data.append(content)

        item = {"name": response.xpath("//*[@id='title']/text()").extract_first(),
                "url": response.url,
                "area": response.xpath("//*[@id='title']/text()").extract_first()[8:10],
                "data": data}
        print('successfully parsed %s' % str(response.url))
        yield item
