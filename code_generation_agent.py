import os
from pathlib import Path
import logging
from typing import List, Dict
from together import Together
import dspy

together_client = Together(api_key="09e8f3626baed45a43804f875e0f422d71ed78320343857bf09ce90bb6d00ae8")

class FeatureImplementer:
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

    def read_test_cases(self, repo_path: Path) -> str:
        """Read the generated test cases from the main directory"""
        test_file_path = repo_path / 'generated_test_cases.py'
        
        if not test_file_path.exists():
            raise FileNotFoundError(f"Test file not found at {test_file_path}")
            
        with open(test_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def analyze_codebase(self, repo_path: Path) -> List[Dict]:
        """Analyze all Python files in the repository"""
        self.logger.info(f"Starting codebase analysis in {repo_path}")
        all_files = []
        
        for file_path in repo_path.rglob("*.py"):
            # Skip test file itself
            if file_path.name == 'generated_test_cases.py' or file_path.name == 'raw_llm_output.py':
                self.logger.debug(f"Skipping file: {file_path.name}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.logger.debug(f"Successfully read file: {file_path}")
                    all_files.append({
                        'path': str(file_path.relative_to(repo_path)),
                        'content': content
                    })
            except Exception as e:
                self.logger.warning(f"Error reading file {file_path}: {str(e)}")
        
        self.logger.info(f"Completed codebase analysis. Found {len(all_files)} Python files")
        return all_files

    def run_test_cases(self, repo_path: Path) -> str:
        """Run the test cases and capture their output"""
        import subprocess
        import sys
        
        self.logger.info("Starting test case execution")
        test_file = repo_path / 'generated_test_cases.py'
        if not test_file.exists():
            self.logger.error(f"Test file not found at {test_file}")
            raise FileNotFoundError(f"Test file not found at {test_file}")
        
        try:
            self.logger.debug(f"Running pytest with command: pytest {str(test_file)} -v")
            # Run pytest with detailed output
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--capture=no"],
                capture_output=True,
                text=True
            )
            self.logger.info("Test execution completed")
            return result.stdout + "\n" + result.stderr
        except Exception as e:
            self.logger.error(f"Error running tests: {str(e)}")
            return str(e)

    def construct_prompt(self, all_files: List[Dict], test_cases: str) -> str:
        """Construct the prompt for the LLM with codebase files and test cases"""
        self.logger.info("Starting prompt construction")
        
        context = ""
        for file in all_files:
            context += f"\nFile: {file['path']}\n```python\n{file['content']}\n```\n"

        # Run test cases and get output
        repo_path = Path(self.temp_dir)
        test_output = self.run_test_cases(repo_path)
        
        self.logger.debug("Adding feature description and test cases to prompt")
        prompt = f"""You are an expert Python developer. Given the following Python codebase and test cases, 
        implement the necessary changes to make the tests pass. The implementation should focus on adding 
        rolling means as features for the demand forecasting model.

        **Requirements:**

        1. **Feature Description:**
        {self.feature_description}

        2. **Test Cases to Satisfy:**
        ```python
        {test_cases}
        ```

        3. **Test Execution Output:**
        The following is the output from running the test cases, showing which tests are failing and need to be fixed:
        ```
        {test_output}
        ```

        4. **Existing Codebase:**
        {context}

        **Implementation Guidelines:**

        1. Implement the necessary changes to make all test cases pass
        2. Follow Python best practices and maintain code readability
        3. Add appropriate comments explaining the implementation
        4. Ensure the implementation aligns with the feature description
        5. Only modify or add what's necessary to implement the rolling means feature
        6. Focus on fixing the failing tests shown in the test execution output

        **Response Format:**
        Provide the complete implementation code for each file that needs to be modified or created. 
        Each file's code should be enclosed in triple backticks with the file path specified.
        Example:
        ```python:path/to/file.py
        # Implementation code here
        ```
        """
        
        self.logger.info("Prompt construction completed")
        return prompt

    def implement_features_with_llm(self, prompt: str) -> Dict:
        """Generate implementation code using the LLM"""
        try:
            self.logger.info("Starting LLM implementation generation")
            self.logger.debug(f"Using model: {self.model}")
            
            # First get the thought process
            self.logger.debug("Generating thought process with DeepSeek-R1")
            # thought = together_client.chat.completions.create(
            #     model="deepseek-ai/DeepSeek-R1",
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=25000,
            #     stop=['</think>']
            # )
            # print(thought.choices[0].message.content)
            # self.logger.debug("Generating implementation with Qwen model")
            # # Add thought process to prompt and generate implementation
            # prompt_with_thought = prompt + f"""
            # Consider this analysis when implementing the feature:
            # <think>
            # {thought.choices[0].message.content}
            # </think>
            # """
            
            lm = dspy.LM("openai/Qwen/Qwen2.5-Coder-32B-Instruct", 
                        api_key=self.together_api_key,
                        api_base="https://api.together.xyz/v1",
                        max_tokens=20000)
            dspy.configure(lm=lm)
            
            # result = lm(prompt_with_thought)[0]
            result = lm(prompt)[0]
            self.logger.info("LLM implementation generation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating implementation: {str(e)}")
            raise

    def write_implementation_files(self, repo_path: Path, implementation: str):
        """Write the generated implementation to files"""
        self.logger.info("Starting to write implementation files")
        #raw output to the file
        with open(repo_path / 'raw_code.py', 'w', encoding='utf-8') as f:
            f.write(implementation)
        # Parse the implementation response to extract file contents
        lines = implementation.split('\n')
        current_file = None
        current_content = []
        files_written = 0
        
        for line in lines:
            if line.startswith('```python:'):
                if current_file:
                    # Write the previous file
                    file_path = repo_path / current_file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_content))
                    self.logger.debug(f"Written file: {current_file}")
                    files_written += 1
                
                # Start new file
                current_file = "new_" + line[10:].strip()
                current_content = []
            elif line.startswith('```'):
                if current_file:
                    # Write the last file
                    file_path = repo_path / current_file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_content))
                    self.logger.debug(f"Written file: {current_file}")
                    files_written += 1
                current_file = None
                current_content = []
            elif current_file:
                current_content.append(line)
        
        self.logger.info(f"Completed writing implementation files. Total files written: {files_written}")

    def implement_features(self):
        """Main method to orchestrate the feature implementation process"""
        try:
            self.setup_logging()
            
            # Use the temp directory created by Agent 1
            repo_name = self.repo_url.split('/')[-1]
            # import time
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.temp_dir = os.path.join(os.getcwd(), f"{repo_name}")
            
            if not os.path.exists(self.temp_dir):
                self.logger.error(f"Directory not found: {self.temp_dir}")
                return
                
            repo_path = Path(self.temp_dir)
            
            # Read test cases and analyze codebase
            test_cases = self.read_test_cases(repo_path)
            all_files = self.analyze_codebase(repo_path)
            
            if not all_files:
                self.logger.warning("No files found for analysis")
                return
            
            # Generate and write implementation
            prompt = self.construct_prompt(all_files, test_cases)
            implementation = self.implement_features_with_llm(prompt)
            self.write_implementation_files(repo_path, implementation)
            
            self.logger.info(f"Implementation completed. Files updated in {self.temp_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in feature implementation process: {str(e)}")
            raise

def main():
    # Example usage
    repo_url = "https://github.com/ChidambaramG/demand_forecasting_XGBoost"
    feature_description = """
    Modify the approach to include rolling means as one of the features when training the model.
    """
    together_api_key = "09e8f3626baed45a43804f875e0f422d71ed78320343857bf09ce90bb6d00ae8"
    model = "together_ai/togethercomputer/Qwen/Qwen2.5-Coder-32B-Instruct"
    
    implementer = FeatureImplementer(repo_url, feature_description, together_api_key, model)
    implementer.implement_features()

if __name__ == "__main__":
    main()
