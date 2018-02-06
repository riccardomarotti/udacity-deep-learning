#!/usr/bin/sh

gunicorn -w 1 -b 127.0.0.1:4000 breedr:app
