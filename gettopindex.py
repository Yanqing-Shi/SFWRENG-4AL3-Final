from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import NoSuchElementException
import requests
from time import sleep
from bs4 import BeautifulSoup
from typing import List
import random
import os

options = webdriver.EdgeOptions()
browser = webdriver.Edge(options=options)

# with open("./thumbnails/catalog.csv", "w") as csv_file:
#     csv_file.write("id, airline, model, manufacture, aov\n")

i = 4

print(f"{i=}")

browser.get(f"https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&category=2&country-location=all&genre=all&keywords-contain=0&keywords-type=all&keywords=&photo-year=all&photographer-group=all&search-type=Advanced&sort-order=0&page={i}")

results = browser.find_elements(By.CLASS_NAME, "result__photoLink")

for i in results:
    index = i.get_attribute("href").split("/")[-1]
    # print(index)
    img = i.find_element(By.CLASS_NAME, "result__photo")
    imgsrc = img.get_attribute("src")
    attrs = img.get_attribute("alt").split(" - ")
    # print(imgsrc)
    # print(attrs)
    img_data = requests.get(imgsrc).content
    with open(f"./thumbnails/{index}.jpg", "wb") as h:
        h.write(img_data)
    with open("./thumbnails/catalog.csv", "a") as c:
        c.write(f"{index},{attrs[2]},{attrs[1]},{" ".join(attrs[1].split(" ")[:-1])},\n")