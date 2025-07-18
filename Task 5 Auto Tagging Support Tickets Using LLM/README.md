# Auto Tagging Support Tickets with LLMs

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/🤗Transformers-4.30+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning system that automatically categorizes customer support tickets using Large Language Models (LLMs) with three different approaches: zero-shot, few-shot, and fine-tuned classification.

## Features

- **Multi-method Approach**: Implements three distinct classification techniques
- **Multi-label Support**: Handles tickets with multiple relevant tags
- **Performance Comparison**: Evaluates different approaches on the same dataset
- **Production-ready**: Includes model saving/loading functionality

## Dataset

The system uses the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) from Hugging Face:

- Contains 20k+ multi-language support tickets
- Each ticket has:
  - Subject line
  - Body text
  - 8 possible tags (tag_1 through tag_8)

## Installation

```bash
!pip install transformers datasets accelerate sentencepiece
