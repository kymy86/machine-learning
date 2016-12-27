#!/usr/bin/python3
import logging

class Logger():

    logging

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def debug(self, str):
        logging.debug(str)



