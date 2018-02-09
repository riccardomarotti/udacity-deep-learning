#!/usr/bin/sh

gunicorn -w 1 --pid /run/gunicorn/pid --bind unix:/run/gunicorn/breedr.socket --pid /run/gunicorn/breedr.pid breedr:app
