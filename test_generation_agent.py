import os
import git
from pathlib import Path
import logging
from typing import List, Dict
import pytest
import pprint
import requests
import dspy
from together import Together

together_client = Together(api_key="09e8f3626baed45a43804f875e0f422d71ed78320343857bf09ce90bb6d00ae8")
class TestCaseGenerator:
    def __init__(self, repo_url: str, feature_description: str, together_api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        self.repo_url = repo_url
        self.feature_description = feature_description
        self.together_api_key = together_api_key
        self.model = model
        self.temp_dir = None
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def clone_repository(self) -> Path:
        """Clone the repository to a new directory in current working directory"""
        repo_name = self.repo_url.split('/')[-1]  # Extract repo name from URL
        self.temp_dir = os.path.join(os.getcwd(), repo_name)
        self.logger.info(f"Cloning repository to {self.temp_dir}")
        
        # Check if directory already exists and handle it
        if os.path.exists(self.temp_dir):
            self.logger.info(f"Directory {self.temp_dir} already exists. Adding timestamp suffix.")
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.temp_dir = f"{self.temp_dir}_{timestamp}"
        
        git.Repo.clone_from(self.repo_url, self.temp_dir)
        return Path(self.temp_dir)

    def analyze_codebase(self, repo_path: Path) -> List[Dict]:
        """
        Analyze all Python files in the repository
        Returns a list of dictionaries containing file paths and their content
        """
        all_files = []
        
        for file_path in repo_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_files.append({
                        'path': str(file_path.relative_to(repo_path)),
                        'content': content
                    })
            except Exception as e:
                self.logger.warning(f"Error reading file {file_path}: {str(e)}")
                
        return all_files

    def construct_prompt(self, all_files: List[Dict]) -> tuple[str, str]:
        """Construct the prompt for the LLM with all codebase files"""

        context = ""
        files_by_dir = {}
        for file in all_files:
            dir_name = str(Path(file['path']).parent)
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(file)
        
        # Add files to prompt, organized by directory
        for dir_name, files in files_by_dir.items():
            context += f"\nDirectory: {dir_name}\n"
            for file in files:
                context += f"\nFile: {file['path']}\n```python\n{file['content']}\n```\n"

        prompt =  f"""You are an expert Python developer specializing in test-driven development (TDD).  Given the 
        following Python code representing a feature implementation, generate comprehensive test cases using pytest.  
        The tests should be designed to thoroughly cover all possible scenarios, 
        including positive cases, negative cases, edge cases, and boundary conditions.  
        Prioritize test coverage and ensure that the tests are robust and reliable.

        **Requirements:**

        1. **pytest Fixtures:** Utilize pytest fixtures effectively to manage test dependencies and setup/teardown operations.  
        Define fixtures for any shared resources or objects needed by the tests.

        2. **Comprehensive Coverage:**  Include a wide range of test cases to cover different input values, data types, 
        and execution paths.  Consider:
            * **Positive Cases:** Valid inputs and expected outputs.
            * **Negative Cases:** Invalid inputs, error conditions, and exception handling. Test for expected exceptions and their types.
            * **Edge Cases:** Boundary conditions, minimum/maximum values, empty inputs, and special cases.
            * **Data Types:** Test with various data types (e.g., integers, floats, strings, lists, dictionaries, None values) where applicable.

        3. **Clear Explanations:**  Add clear and concise comments within each test function to explain its purpose and the specific scenario it covers.  Describe the expected behavior and any relevant edge cases.

        4. **Python Testing Best Practices:** Adhere to Python testing best practices, including meaningful test names, proper use of assertions (e.g., `assert`, `pytest.raises`), and appropriate test structure.

        5. **Complete and Runnable Code:** The generated test code must be complete, self-contained, and runnable without any modifications.  Include all necessary imports and dependencies.  Do not leave any placeholder code or comments indicating missing parts.  The tests must be able to be executed directly using `pytest`.

        6. **No Hallucinations:** Do not invent or assume any functionality that is not explicitly described in the provided code.  Base your tests solely on the given implementation.

        7. **Focus on Functionality:**  The tests should focus on verifying the core functionality of the code.  Avoid testing implementation details that are not relevant to the external behavior.

        8. **Referencing other files:** If the test case requires data from other files, make sure to reference the other files in the test case. And make sure that you are using it
        

        **Code and Feature Description:**

        ```Existing code and files:
        {context}
        ```

        **Feature Description:**

        {self.feature_description}

        **Response Format:**

        The response should contain only the complete, runnable Python test code enclosed within triple backticks 
        (```python ... ```).  Do not include any other text, explanations, or comments outside the code block.
        """
        # pprint.pprint(prompt)
        return prompt

    def generate_tests_with_llm(self, prompt: str) -> Dict:
        """Make POST request to Together API and get the generated tests"""
        try:
            # thought = together_client.chat.completions.create(
            #     model="deepseek-ai/DeepSeek-R1",
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=25000,
            #     stop=['</think>']
            # )
            # print(thought.choices[0].message.content)
            # Make POST request to DeepSeek-R1 using Together API

            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [
                    {"role": "user", "content": prompt + "** Generate the throught process for the given prompt**"}
                ],
                "max_tokens": 25000,
                "stop": ['</think>']
            }

            response = requests.post("https://api.together.xyz/inference", headers=headers, json=data)
            response.raise_for_status()
            thought_result = response.json()["choices"][0]["message"]["content"]
            print(thought_result)


            # Make POST request to DeepSeek-R1
            thought_lm = dspy.LM("openai/Qwen/Qwen2.5-Coder-32B-Instruct", api_key=self.together_api_key, 
                         api_base="https://api.together.xyz/v1", max_tokens=25000)
            
            dspy.configure(lm=thought_lm)
            thought_result = thought_lm(prompt)[0]
            print(thought_result)

            prompt_with_thought = prompt + f""" 
            The following is the thought process of the Python Developer. Use the given thought process 
            to generate the test cases.
            <think>
            {thought_result}
            </think>
            """
            lm = dspy.LM("openai/Qwen/Qwen2.5-Coder-32B-Instruct", api_key=self.together_api_key, 
                         api_base="https://api.together.xyz/v1", max_tokens=22000)
            dspy.configure(lm=lm)
            return lm(prompt_with_thought)[0]
            
        except Exception as e:
            self.logger.error(f"Error generating tests: {str(e)}")
            raise

    def write_test_file(self, repo_path: Path, test_data: str) -> Path:
        """Write the generated tests to a file"""
        # Write complete output to raw file
        raw_output_path = repo_path / 'raw_llm_output.py'
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(test_data)
            
        test_file_path = repo_path / 'generated_test_cases.py'
        
        # Extract code between ```python and ``` markers
        code_blocks = []
        lines = test_data.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip() == '```python':
                in_code_block = True
                continue
            elif line.strip() == '```':
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = False
                continue
                
            if in_code_block:
                current_block.append(line)
        
        # Write only the extracted code blocks
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(code_blocks))
            
        return test_file_path

    def run_tests(self, test_file_path: Path):
        """Run the generated tests"""
        self.logger.info(f"Running tests from {test_file_path}")
        pytest.main([str(test_file_path), "-v"])

    def generate_and_run_tests(self):
        """Main method to orchestrate the test generation process"""
        try:
            self.setup_logging()
            repo_path = self.clone_repository()
            all_files = self.analyze_codebase(repo_path)
            
            if not all_files:
                self.logger.warning("No files found for analysis")
                return
            
            prompt = self.construct_prompt(all_files)
            test_data = self.generate_tests_with_llm(prompt)
            # pprint.pprint(test_data)
            test_file_path = self.write_test_file(repo_path, test_data)
            self.run_tests(test_file_path)
            
        except Exception as e:
            self.logger.error(f"Error in test generation process: {str(e)}")
            raise
        finally:
            if self.temp_dir:
                self.logger.info(f"Generated tests can be found in {self.temp_dir}")

def main():
    # Example usage
    repo_url = "https://github.com/ChidambaramG/demand_forecasting_XGBoost"
    feature_description = """
    Modify the approach to include rolling means as on of the features when training the model.
    """
    together_api_key = "09e8f3626baed45a43804f875e0f422d71ed78320343857bf09ce90bb6d00ae8"
    model = "together_ai/togethercomputer/Qwen/Qwen2.5-Coder-32B-Instruct"
    
    generator = TestCaseGenerator(repo_url, feature_description, together_api_key)
    generator.generate_and_run_tests()

if __name__ == "__main__":
    main()