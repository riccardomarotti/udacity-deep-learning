#!/usr/bin/sh

gunicorn -b 127.0.0.1:4000 breedr:app
