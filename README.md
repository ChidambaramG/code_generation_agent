# 🤖 Code Generation Agent

A sophisticated AI-powered code generation system that uses a unique "thought-first" approach with large language models to generate high-quality code implementations and test cases.

## 🌟 Key Features

- 🧠 Two-stage thinking process using specialized LLMs
- 🔄 Automated test case generation
- 💻 Intelligent code implementation
- 🔍 Comprehensive codebase analysis
- 📝 Detailed logging and documentation

## 🎯 Core Components

### 1. Test Case Generator
- Clones target repository
- Analyzes existing codebase
- Generates comprehensive test cases using pytest
- Implements test fixtures and edge cases

### 2. Feature Implementer
- Implements new features based on test cases
- Ensures code quality and best practices
- Validates implementation against tests
- Maintains existing codebase integrity

## 🧠 Unique "Think-First" Approach

This project implements a novel two-stage thinking process:

1. **Thought Generation** 🤔
```python
# First, generate deep analysis using DeepSeek-V3
thought = together_client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[...],
    stop=['</think>']
)
```

2. **Code Implementation** ⌨️
```python
# Then, use Qwen Coder with the generated thoughts
prompt_with_thought = prompt + f"""
Consider this analysis when implementing:
<think>
{thought_result}
</think>
"""
```

### Why This Approach Works
- 🎯 Better problem understanding
- 🔍 More thorough solution analysis
- 💡 Improved code quality
- 🐛 Fewer bugs and edge cases missed

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Together AI API key
- Git

### Installation
```bash
git clone https://github.com/yourusername/code_generation_agent
cd code_generation_agent
pip install -r requirements.txt
```

### Usage
```python
from code_generation_agent import TestCaseGenerator, FeatureImplementer

# Initialize and run test generation
generator = TestCaseGenerator(
    repo_url="your_repo_url",
    feature_description="your_feature_description",
    together_api_key="your_api_key"
)
generator.generate_and_run_tests()

# Implement features
implementer = FeatureImplementer(
    repo_url="your_repo_url",
    feature_description="your_feature_description",
    together_api_key="your_api_key"
)
implementer.implement_features()
```

## 📊 Model Configuration

The system uses specific LLM models for different tasks:
- 🤔 **Thought Generation**: DeepSeek-V3
- 💻 **Code Generation**: Qwen2.5-Coder-32B-Instruct

## 🔐 Security

- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue in the repository.