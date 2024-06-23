#!/usr/bin/python3

class csvException(Exception):
       def __init__(self, message):
        super().__init__(message)