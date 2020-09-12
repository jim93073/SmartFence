# import urllib.request
# url = "http://download.thinkbroadband.com/10MB.zip"
#
#
# def show_progress(count, block_size, total_size):
#     print(count, block_size, total_size)
#
#
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
# urllib.request.install_opener(opener)
# urllib.request.urlretrieve(url, "10M.zip", show_progress)
from .. import utils

utils.download("http://download.thinkbroadband.com/10MB.zip", "10MB.zip")
