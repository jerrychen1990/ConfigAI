#!/bin/bash
echo 'run unit tests'
echo "executing python -m unittest discover -s tests/ -p 'test_*.py'"
python -m unittest discover -s tests/ -p 'test_*.py'