# -*- coding: utf-8 -*-

# Scrapy settings for scrapy_autohome project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'scrapy_autohome'

SPIDER_MODULES = ['scrapy_autohome.spiders']
NEWSPIDER_MODULE = 'scrapy_autohome.spiders'
# 绕过robots.txt
ROBOTSTXT_OBEY = False

#记录日志
LOG_FILE = "scrapy_autohome_log.log"

# 保存文件编码类型
FEED_EXPORT_ENCODING = 'GBK'

# 伪装chrome
# USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'

# #DOWNLOADER_MIDDLEWARES
# DOWNLOADER_MIDDLEWARES = {
#         'scrapy.contrib.downloadermiddleware.httpproxy.HttpProxyMiddleware':301,
#     }

# IPPOOL=[
#     {"ipaddr":"27.220.54.6:9000"},
#     {"ipaddr":"27.206.176.45:9000"},
#     {"ipaddr":"222.94.196.203:3128"},
#     {"ipaddr":"27.206.76.53:9000"},
#     {"ipaddr":"27.220.122.96:9000"},
#     {"ipaddr":"42.59.109.218:9999"},
#     {"ipaddr":"27.206.181.35:9000"},
#     {"ipaddr":"114.99.13.164:1133"}
# ]

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'scrapy_autohome (+http://www.yourdomain.com)'



# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 2

# Configure a delay for requests for the same website (default: 0)
# See http://scrapy.readthedocs.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
# DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'scrapy_autohome.middlewares.ScrapyAutohomeSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
# DOWNLOADER_MIDDLEWARES = {
#    'scrapy_autohome.middlewares.SeleniumMiddleware': 300,
# }
RANDOM_UA_TYPE = 'random'   ##random    chrome

DOWNLOADER_MIDDLEWARES = {
    'scrapy_autohome.MidWare.user_agent_middlewares.RandomUserAgentMiddlware': 543,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware':None,
    # 'scrapy_autohome.middlewares.MyproxiesSpiderMiddleware':300,
    # 'scrapy_autohome.middlewares.IPPOOL':545,
}

# Enable or disable extensions
# See http://scrapy.readthedocs.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See http://scrapy.readthedocs.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'scrapy_autohome.pipelines.ScrapyAutohomePipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See http://doc.scrapy.org/en/latest/topics/autothrottle.html
# AUTOTHROTTLE_ENABLED = True
# The initial download delay
# AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
# AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
