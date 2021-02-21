#!/bin/bash

# Run tests for 48khz
echo "Tests for 48000Hz wave"
ln -sf $PWD/tests/test_files/test_file_48000Hz.wav $PWD/test.wav
pytest --cache-clear -n 8 -q --cov=./spafe tests/test_*.py

# Run tests for 44.1khz
echo "Tests for 441000Hz wave"
ln -sf $PWD/tests/test_files/test_file_44100Hz.wav $PWD/test.wav
pytest --cache-clear -n 8 -q --cov=./spafe tests/test_*.py

# Run tests for 32khz
echo "Tests for 32000Hz wave"
ln -sf $PWD/tests/test_files/test_file_32000Hz.wav $PWD/test.wav
pytest --cache-clear -n 8 -q --cov=./spafe tests/test_*.py

# Run tests for 8khz
echo "Tests for 8000Hz wave"
ln -sf $PWD/tests/test_files/test_file_8000Hz.wav $PWD/test.wav
pytest --cache-clear -n 8 -q --cov=./spafe tests/test_*.py

# Run tests for 16khz
echo "Tests for 16000Hz wave"
ln -sf $PWD/tests/test_files/test_file_16000Hz.wav $PWD/test.wav
pytest --cache-clear -n 8 -q --cov=./spafe tests/test_*.py
