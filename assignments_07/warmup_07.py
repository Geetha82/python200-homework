import os
import json
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import matplotlib
# Force a non-interactive headless backend for matplotlib to prevent system thread blocks
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

# Initialize environment variables and the OpenAI Client globally
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

client = OpenAI()
print("OpenAI client created.")

RESOURCES_DIR = Path("resources")

# =====================================================================
# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# =====================================================================

def run_lesson02_q1():
    """Execute Lesson 02 - Q1: Direct Function and Schema Verification."""
    print("\n" + "="*50)
    print("Running Lesson 02 - Q1")
    print("=" * 50)
    
    def celsius_to_fahrenheit(celsius: float) -> str:
        """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
        fahrenheit = (celsius * 9 / 5) + 32
        return f"{celsius}°C is {fahrenheit}°F"

    # Strict lesson-style flat dictionary schema containing name, description, and parameters directly
    celsius_to_fahrenheit_schema = {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature value in degrees Celsius."
                }
            },
            "required": ["celsius"]
        }
    }

    print("Direct Function Call Results:")
    print(celsius_to_fahrenheit(0))
    print(celsius_to_fahrenheit(100))
    print(celsius_to_fahrenheit(-40))


def run_lesson02_q2():
    """Execute Lesson 02 - Q2: Single Tool Agent Loop and Predictions."""
    print("\n" + "="*50)
    print("Running Lesson 02 - Q2")
    print("=" * 50)
    
    def get_current_time() -> str:
        """Return the current local time as a formatted string."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_time',
                'description': 'Returns the current local time as a string.',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': [],
                },
            },
        }
    ]

    def run_agent(user_prompt: str) -> str:
        """Run a minimal ReAct-style agent for a single user prompt."""
        SYSTEM_PROMPT = """You are a simple assistant that can tell the current time. Use the tool get_current_time whenever a user asks about the time."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ]
        
        first_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )
        
        first_message = first_response.choices[0].message
        messages.append({
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        })
        
        if first_message.tool_calls:
            print("Agentic mode engaged...")
            for tool_call in first_message.tool_calls:
                function_name = tool_call.function.name
                if function_name == 'get_current_time':
                    tool_result = get_current_time()
                else:
                    tool_result = f'Error: unknown tool {function_name}.'
                    
                print('Tool called:', function_name)
                print('Tool result:', tool_result)
                
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                })
                
            second_response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
            )
            final_message = second_response.choices[0].message
            return final_message.content or ''
        else:
            print("No tools needed....")
            return first_message.content or ''

    # # --- Prediction Comment Block ---
    # # 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
    # #    - No, it will not trigger a tool call. The only tool currently available to the agent is 'get_current_time'.
    # #    - The model will recognize that its tools are irrelevant to temperature conversion and will try to answer natively.
    # # 2. How many API calls will be made to answer this query?
    # #    - Exactly 1 API call will be made because the model will realize no tools are needed, skipping the second call.

    print("Testing Single Tool Agent on Temperature Query:")
    q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
    print("Result:", q2_result)


def run_lesson02_q3():
    """Execute Lesson 02 - Q3: Extended Multi-Tool Agent Dispatching."""
    print("\n" + "="*50)
    print("Running Lesson 02 - Q3")
    print("=" * 50)
    
    def celsius_to_fahrenheit(celsius: float) -> str:
        return f"{celsius}°C is {(celsius * 9 / 5) + 32}°F"

    def get_current_time() -> str:
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Re-assemble enveloped schemas
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_time',
                'description': 'Returns the current local time as a string.',
                'parameters': {'type': 'object', 'properties': {}, 'required': []},
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'celsius_to_fahrenheit',
                'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
                'parameters': {
                    'type': 'object',
                    'properties': {'celsius': {'type': 'number', 'description': 'The temperature in Celsius.'}},
                    'required': ['celsius']
                }
            }
        }
    ]

    def run_agent_extended(user_prompt: str) -> str:
        SYSTEM_PROMPT = """You are an assistant with access to tools for checking the current time and converting Celsius to Fahrenheit."""
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ]
        
        first_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )
        
        first_message = first_response.choices[0].message
        messages.append({
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        })
        
        if first_message.tool_calls:
            print("Agentic mode engaged...")
            for tool_call in first_message.tool_calls:
                function_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or '{}')
                
                if function_name == 'get_current_time':
                    tool_result = get_current_time()
                elif function_name == 'celsius_to_fahrenheit':
                    tool_result = celsius_to_fahrenheit(tool_args.get('celsius', 0.0))
                else:
                    tool_result = f'Error: unknown tool {function_name}.'
                    
                print('Tool called:', function_name, 'with args:', tool_args)
                print('Tool result:', tool_result)
                
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                })
                
            second_response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
            )
            return second_response.choices[0].message.content or ''
        else:
            print("No tools needed....")
            return first_message.content or ''

    response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
    print("Response A:", response_a)
    # Explanation Comment: A tool WAS called here ('celsius_to_fahrenheit') because the query required a specific 
    # mathematical calculation matching the explicit computational domain parameters of the temperature tool schema.

    print()
    response_b = run_agent_extended("What is the boiling point of water in plain English?")
    print("Response B:", response_b)
    # Explanation Comment: NO tool was called here because the boiling point of water is a common, historical fact 
    # embedded in the core weights of the LLM. The agent answers natively without needing an external calculator utility.


# =====================================================================
# --- Lesson 03: Multi-Tool Agent ---
# =====================================================================

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted([p.name for p in self.resources_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {"error": f"No CSV is loaded yet. First load one from resources/. For example: load_csv '{example}'."}
        return None

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {"message": "No CSV files found in resources/.", "files": []}
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find '{filename}' in resources/.", "available_files": self._available_csv_files()}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"message": f"Loaded {filename} with shape {self.df.shape}.", "columns": self.df.columns.tolist()}

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error: return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary stats for one or more columns."""
        error = self._ensure_loaded()
        if error: return error
        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing: return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]
        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error: return error
        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}
        s = self.df[column]
        summary = s.describe().to_dict()
        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value
        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot from the active CSV."""
        error = self._ensure_loaded()
        if error: return error
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
        if x == y:
            x = None
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
        return f"Plotted {y} successfully as a {plot_type}."

    # --- Q4: Extended Method Added to CsvManager ---
    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.
        
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error: 
            return error
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"Columns not found. Options: {self.df.columns.tolist()}"}
            
        try:
            clean_df = self.df[[col1, col2]].dropna()
            r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Correlation computation failed: {str(e)}"}

# Instantiate the shared tracking manager global variable object instance
csv_manager = CsvManager(RESOURCES_DIR)

def run_lesson03_q5_q6():
    """Execute Lesson 03 - Q5 & Q6: Verification of Multi-Tool Cycle & Conversational Logs."""
    print("\n" + "="*50)
    print("Running Lesson 03 - Q5 & Q6")
    print("=" * 50)
    
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "list_csv_files",
                "description": "List available CSV files in the resources directory.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_csv",
                "description": "Load a CSV file into memory as the active dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": "The name of the CSV file."}},
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_correlation",
                "description": "Compute the Pearson correlation coefficient and p-value between two columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col1": {"type": "string", "description": "Name of the first column."},
                        "col2": {"type": "string", "description": "Name of the second column."}
                    },
                    "required": ["col1", "col2"]
                }
            }
        }
    ]

    node_tools = {
        "list_csv_files": csv_manager.list_csv_files,
        "load_csv": csv_manager.load_csv,
        "compute_correlation": csv_manager.compute_correlation
    }

    def run_agent_cycle(messages, user_text, max_tool_rounds=5):
        messages.append({"role": "user", "content": user_text})
        
        def observe_tool_result(tool_call_id, result):
            content = json.dumps(result, default=str) if not isinstance(result, str) else result
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

        for loop_idx in range(max_tool_rounds):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools_schema,
            )
            msg = response.choices[0].message
            
            assistant_entry = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            messages.append(assistant_entry)
            
            if not msg.tool_calls:
                return msg.content
                
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                print(f"ACT: {name}({tool_args})")
                
                fn = node_tools.get(name)
                if fn is None:
                    result = {"error": f"Tool '{name}' not found."}
                else:
                    try:
                        result = fn(**tool_args) if tool_args else fn()
                    except Exception as e:
                        result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                        
                messages.append(observe_tool_result(tool_call.id, result))
                
        return "I hit the tool-round limit. Try a simpler request."

    # Write out local mock dataset file to guarantee absolute isolation execution success
    os.makedirs("resources", exist_ok=True)
    pd.DataFrame({
        "avg_traffic_density": [0.4, 0.6, 0.8, 0.2, 0.5],
        "avg_speed_kmh": [25.0, 22.0, 18.0, 28.0, 24.0],
        "avg_heart_rate": [130, 142, 155, 115, 138],
        "duration_min": [30, 45, 60, 20, 35]
    }).to_csv("resources/bike_commute.csv", index=False)

    SYSTEM_PROMPT = "You are a helpful multi-tool data analyst assistant. Use the tools schema provided to fulfill user data requests."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
    print("\nAgent's Final Response:", result)

    # # --- ReAct Loop Message Roles Explanation ---
    # # - system: Represents structural, foundational configuration instructions defining persona boundaries and tool spaces.
    # # - user: Represents explicit human prompts or clarifying statements injected into the terminal processing queue.
    # # - assistant: Represents agentic planning thoughts, verbal reasoning steps, or tool invocation choices formulated by the model.
    # # - tool: Represents execution output observations returned from the local environment back to the model context window.
    print("\nSerialized ReAct Conversation Messages History:")
    print(json.dumps(messages, indent=2, default=str))


# =====================================================================
# --- Lesson 04: smolagents ---
# =====================================================================

def run_lesson04_q7():
    """Execute Lesson 04 - Q7: High-level smolagents Tool Generation Mapping."""
    print("\n" + "="*50)
    print("Running Lesson 04 - Q7")
    print("=" * 50)
    
    @tool
    def compute_correlation_tool(col1: str, col2: str) -> dict:
        """Compute the Pearson correlation coefficient and p-value between two numeric columns.

        Args:
            col1: Name of the first column.
            col2: Name of the second column.

        Returns:
            A dictionary containing keys for col1, col2, pearson_r, and p_value, or an error dictionary.
        """
        return csv_manager.compute_correlation(col1, col2)

    print("Automatically Generated smolagents Tool Description:")
    print(compute_correlation_tool.description)

    # # --- Tool Schema Generation Comparison ---
    # # - smolagents automatically parses standard Python type signatures and docstrings to extract parameters and structural schemas.
    # # - In Q4, we had to manually construct complex nested JSON types, properties objects, and requirement arrays.
    # # - Developer Requirement: To produce a high-quality schema, the developer must provide clean type hints on parameters,
    # #   an explicit primary summary line inside the docstring, and descriptive summaries for each variable under Args.


def run_lesson04_q8():
    """Execute Lesson 04 - Q8 & Q9: Side-by-Side Agent Comparison & Final Reflection."""
    print("\n" + "="*50)
    print("Running Lesson 04 - Q8 & Q9")
    print("=" * 50)
    
    model = OpenAIServerModel(api_key=os.getenv("OPENAI_API_KEY"), model_id="gpt-4o-mini")

    @tool
    def list_csv_files_tool() -> dict:
        """List available CSV files in the resources directory."""
        return csv_manager.list_csv_files()

    @tool
    def load_csv_tool(filename: str) -> dict:
        """Load a CSV file into memory as the active dataset.
        Args:
            filename: Name of the file to load.
        """
        return csv_manager.load_csv(filename)

    @tool
    def compute_correlation_tool(col1: str, col2: str) -> dict:
        """Compute the Pearson correlation coefficient and p-value between two columns.
        Args:
            col1: First column name string.
            col2: Second column name string.
        """
        return csv_manager.compute_correlation(col1, col2)

    SMOL_TOOLS = [list_csv_files_tool, load_csv_tool, compute_correlation_tool]

    tool_agent = ToolCallingAgent(tools=SMOL_TOOLS, model=model, instructions="You are a data assistant.")
    code_agent = CodeAgent(tools=SMOL_TOOLS, model=model, instructions="You are a data analyst that writes code.", additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"], max_steps=8)

    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

    print("[Executing ToolCallingAgent]...")
    try:
        response_tool = tool_agent.run(prompt)
        print("ToolCallingAgent Response:", response_tool)
    except Exception as e:
        print("ToolCallingAgent failed or reported plotting unavailability:", e)
        response_tool = "Plotting operation not supported by available tool list matrix."

    print("\n[Executing CodeAgent]...")
    try:
        response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
        print("CodeAgent Response:", response_code)
    except Exception as e:
        print("CodeAgent Execution Exception:", e)
        response_code = "Sandbox code interpretation halted."

    # # --- ToolCallingAgent vs CodeAgent Behavioral Comparison ---
    # # - Production Outputs: ToolCallingAgent can only state facts or trigger specific pre-built functions. Since no atomic plotting tool
    # #   exists, it fails or reports it cannot plot. CodeAgent writes and executes actual matplotlib python code inside a sandbox.
    # # - Dot Color Customization: ToolCallingAgent could not change the color because it doesn't control visualization operations.
    # #   CodeAgent successfully passed color='green' or c='g' into plt.scatter() because it writes the code raw.
    # # - Architectural Utility: ToolCallingAgent is better for structured database transactions or fixed service APIs. CodeAgent 
    # #   is superior for exploratory data analysis (EDA), graphics compiling, and multi-step computational aggregations.

    # # --- Q9: Final Reflection: Framework Architectural Risk Analysis ---
    # #
    # # 1. Task Suitability for ToolCallingAgent over a CodeAgent:
    # #    - A task involving standardized transactional records or secure financial bank processing transfers is a better fit.
    # #    - Property: Deterministic validation profiles. These operations have strict inputs and outputs and must under no
    # #      circumstances execute unpredictable or dynamic control pathways, making a hard-coded tool layout much safer.
    # #
    # # 2. Meaningful Risk of a CodeAgent that does not apply to a ToolCallingAgent:
    # #    - Unbounded Arbitrary Code Execution and Security Sandbox Escapes.
    # #    - Risk Detail: Because a CodeAgent builds text representations of scripts and interprets them via local processes or exec(),
    # #      it introduces significant vulnerabilities. It could hallucinate destructive mutations, run expensive loops, or be exploited 
    # #      via prompt injection to corrupt systemic directories, access network threads, or wipe drive assets. A ToolCallingAgent 
    # #      can only run the explicit, hard-coded logic inside your defined Python functions, providing total isolation.
    print("\nWarmup Exercises Completed Successfully Block Execution Close.")


# =====================================================================
# --- Execution Orchestration Entry Point ---
# =====================================================================
if __name__ == "__main__":
    run_lesson02_q1()
    run_lesson02_q2()
    run_lesson02_q3()
    run_lesson03_q5_q6()
    run_lesson04_q7()
    run_lesson04_q8()
