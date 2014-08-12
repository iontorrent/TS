# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf import settings
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait


#determine the WebDriver module. default to Firefox
try:
    web_driver_module = settings.SELENIUM_WEBDRIVER
except AttributeError:
    from selenium.webdriver.firefox import webdriver as web_driver_module


class CustomWebDriver(web_driver_module.WebDriver):
    """Our own WebDriver with some helpers added"""

    def find_css(self, css_selector):
        """Shortcut to find elements by CSS. Returns either a list or singleton"""
        elems = self.find_elements_by_css_selector(css_selector)
        found = len(elems)
        if found == 1:
            return elems[0]
        elif not elems:
            raise NoSuchElementException(css_selector)
        return elems

    def wait_for_css(self, css_selector, timeout=7):
        """ Shortcut for WebDriverWait"""
        try:
            return WebDriverWait(self, timeout).until(lambda driver : driver.find_css(css_selector))
        except:
            self.quit()

    def wait_for_ajax(self, timeout=10000):
        try:
            return WebDriverWait(self, timeout).until(lambda driver : 0 == driver.execute_script("return jQuery.active"))
        except:
            self.quit()
