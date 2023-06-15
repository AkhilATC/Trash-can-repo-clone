import scrapy
import re


class HcpcsDataScraper(scrapy.Spider):
    name = "Hcpcs_data_scraper"
    allowed_domains = ['https://www.hcpcsdata.com/Codes']
    start_urls = [f"https://www.hcpcsdata.com/Codes/{x}" for x in "ABCDEGHJKLMPQRSTUV"]

    # depricated
    def preprocess_td_elements(self, element):

        element.replace("\r\n","")
        td_pattern = r'(?i)<td.*?>([^<]+)</td.*?>'
        patterns = re.findall(td_pattern, element)
        print(patterns)
        patterns = list(filter(None, patterns))
        return "".join(patterns)

    def preprocess(self,elem):
        elem = elem.replace("\r\n", "").strip()
        return elem if elem else None

    def parse(self, response, **kwargs):
        print(response.url)
        data = list(filter(None,[self.preprocess(x) for x in response.css('tr.clickable-row td::text').getall() if x != ""]))
        title = [head for head in response.css('a.identifier::text').getall()]
        if len(data) != len(title):
            raise Exception("parser error")
        for index,value in enumerate(data):
            yield {
                'Group': response.css('div.body-content h1::text').getall(),
                'catagory': response.css('div.body-content h5::text').getall(),
                'long discription': value,
                'code': title[index]
            }




