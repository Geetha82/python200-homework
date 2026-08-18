import os
import json
from datetime import datetime
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import matplotlib
# Force a non-interactive headless backend for matplotlib to prevent macOS terminal freezes
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, tool

# =====================================================================
# --- Environment Setup & Client Initialization ---
# =====================================================================
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
print("OpenAI client created.")

RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)

# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1 
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema dictionary describing this function to an LLM
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
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
}

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Baseline tools list from the lesson
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

def run_lesson02_q1():
    print("\n--- Lesson 02 - Q1 ---")
    print("Calling function directly with 0, 100, and -40:")
    print(celsius_to_fahrenheit(0))
    print(celsius_to_fahrenheit(100))
    print(celsius_to_fahrenheit(-40))

# Q2
def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt using get_current_time."""
    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time. Use the tool get_current_time whenever a user asks about the time."""
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )
    print("First response received from model...")
    print(first_response)
    
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
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)
        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")
        return first_message.content or ''

def run_lesson02_q2():
    print("\n--- Lesson 02 - Q2 ---")
    # --- Prediction Comment Block ---
    # 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
    #    No, because the only available tool in the tools list is 'get_current_time'. The model will recognize
    #    that checking the local time is completely irrelevant to a temperature conversion request.
    # 2. How many API calls will be made to answer this query?
    #    Exactly 1 API call will be made because the model will not request a tool, bypassing the secondary completion block.
    
    result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
    print("Result:", result)
    print("Comment: The prediction was correct. The agent answered natively without invoking any tools in 1 step.")

# Q3
def run_agent_extended(user_prompt: str) -> str:
    """Extended minimal ReAct agent supporting both get_current_time and celsius_to_fahrenheit tools."""
    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time and convert temperatures. Use the tools whenever appropriate."""
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    
    # Extended tools array containing both schemas
    extended_tools = [tools[0], celsius_to_fahrenheit_schema]
    
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=extended_tools,
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
            args = json.loads(tool_call.function.arguments or "{}")
            
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit(args.get("celsius"))
            else:
                tool_result = f'Error: unknown tool {function_name}.'
                
            print('Tool called:', function_name, 'with args:', args)
            print('Tool result:', tool_result)
            
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': function_name,
                'content': tool_result,
            })
            
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        return second_response.choices[0].message.content or ''
    else:
        print("No tools needed....")
        return first_message.content or ''

def run_lesson02_q3():
    print("\n--- Lesson 02 - Q3 ---")
    response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
    print("Response A:", response_a)
    # Comment: A tool WAS called here because the user explicitly requested a temperature conversion,
    # which matched the capabilities and parameters defined inside the celsius_to_fahrenheit tool schema.
    
    print()
    response_b = run_agent_extended("What is the boiling point of water in plain English?")
    print("Response B:", response_b)
    # Comment: NO tool was called here because explaining the boiling point of water is a common pre-trained fact.
    # The model handles the query natively using its own knowledge base without executing any mathematical calculations.

# --- Lesson 03: Multi-Tool Agent ---

class CsvManager:
    """Class manager tracking dataset frames and operational configurations from the lesson."""
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None
        
        # Self-contained safety: bootstrap 'bike_commute.csv' locally if missing so testing never crashes
        mock_file = self.resources_dir / "bike_commute.csv"
        if not mock_file.exists():
            pd.DataFrame({
                "avg_traffic_density": [0.3, 0.5, 0.7, 0.9, 0.4],
                "avg_speed_kmh": [28.4, 22.1, 18.5, 12.0, 25.3],
                "avg_heart_rate": [125, 138, 145, 160, 130],
                "duration_min": [30, 35, 40, 50, 25]
            }).to_csv(mock_file, index=False)

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
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
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
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

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
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
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
            
        title_csv = self.csv_name or "current CSV"
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.savefig("outputs/line_plot.png")
            plt.close()
            return f"Plotted {y} vs row index as a line plot."
            
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.savefig("outputs/scatter_plot.png")
        plt.close()
        return f"Plotted {y} vs {x} as a {plot_type}."

    # Q4
    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.
        
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error: return error
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
            return {"error": f"Calculation failed: {str(e)}"}

print("Class defined")
csv_manager = CsvManager(RESOURCES_DIR)

# System Prompt from the lesson
SYSTEM_PROMPT = "You are a data analyst assistant. Use your tools sequentially to analyze data files stored in resources/."

# Setup schemas and tools mapping for Lesson 03
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in resources/.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from resources/ and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The target CSV file name."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First numeric column header name."},
                    "col2": {"type": "string", "description": "Second numeric column header name."}
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

# Verbatim Lesson Agent Cycle Implementation
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """Run through one react-agent loop using a simple tool-using agent."""
    messages.append({"role": "user", "content": user_text})
    
    def observe_tool_result(tool_call_id, result):
        """Return a tool's return value as a message that can be appended to the LLMs conversation history."""
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        return tool_message
        
    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
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
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            messages.append(observe_tool_result(tool_call.id, result))
            
    return "I hit the tool-round limit. Try a simpler request."

# Q5 & Q6
def run_lesson03_q5_q6():
    print("\n--- Lesson 03 - Q5 ---")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
    print("Final Agent Response:", result)
    
    print("\n--- Lesson 03 - Q6 ---")
    # Comment identifying what each role represents in the ReAct loop:
    # - 'system': Injects global persona guidelines, contextual instructions, and structural descriptions of all tools.
    # - 'user': Captures the user's specific query text prompt to initialize or continue the agentic loop.
    # - 'assistant': Tracks the inner thought processes, planning text, and functional tool payload invocations of the LLM.
    # - 'tool': Supplies the local system's operational or empirical data responses back into the agent context window.
    print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---

# Explicit decorated smolagents tool wrappers matching your specifications
@tool
def list_csv_files_tool() -> dict:
    """List available CSV files in resources/.
    
    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv_tool(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
    
    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".
        
    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns_tool() -> list[str] | dict:
    """Return column names for the currently loaded CSV.
    
    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns_tool(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.
    
    Args:
        columns: Column names to summarize. If None, summarizes all columns.
        
    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column_tool(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.
    
    Args:
        column: The name of the column to describe.
        
    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)

@tool
def plot_data_tool(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.
    
    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
        
    Returns:
        Generates and saves the plot. Returns a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

# Q7
@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: First numeric column header name.
        col2: Second numeric column header name.

    Returns:
        A dict showing the structural keys of col1, col2, pearson_r, and p_value metrics.
    """
    return csv_manager.compute_correlation(col1, col2)

def run_lesson04_q7():
    print("\n--- Lesson 04 - Q7 ---")
    print(compute_correlation_tool.description)
    # Comment: smolagents automatically compiles standard JSON function schemas by extracting 
    # the python function signatures, docstrings, type hints, and parameter descriptions. 
    # Compared to writing manual schemas in Q4, smolagents reduces manual dict boilerplate. 
    # The developer must write explicit documentation on argument names and types so that 
    # the parser can translate descriptions into valid structural LLM tool profiles.

# Q8
def run_lesson04_q8():
    print("\n--- Lesson 04 - Q8 ---")
    
    model_to_use = "gpt-4o-mini"
    model = OpenAIServerModel(api_key=api_key, model_id=model_to_use)
    
    SYSTEM_PROMPT_L4 = (
        "You are a small data assistant to help analyze files stored in resources/. "
        "Use the available tools to do any work requested (do not guess). "
        "Keep answers short and student-friendly."
    )
    
    CODE_INSTRUCTIONS = (
        "You are a code data analyst assistant. Write custom Python plotting snippets "
        "using matplotlib when charts are requested."
    )
    
    # Matching the exact lesson structures
    TOOLS = [list_csv_files_tool, load_csv_tool, get_columns_tool, summarize_columns_tool, describe_column_tool, plot_data_tool, compute_correlation_tool]
    
    tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT_L4)
    code_agent = CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=CODE_INSTRUCTIONS,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
        max_steps=8,
    )
    
    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
    csv_manager.load_csv("bike_commute.csv")
    
    print("\nRunning ToolCallingAgent...")
    try:
        response_tool = tool_agent.run(prompt)
        print("Response Tool:", response_tool)
    except Exception as e:
        print("Tool Agent Exception:", e)
        
    print("\nRunning CodeAgent...")
    try:
        response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
        print("Response Code:", response_code)
    except Exception as e:
        print("Code Agent Exception:", e)

    # --- Comment Block for Q8 Questions ---
    # What did each agent actually produce?
    # - ToolCallingAgent could not customize individual plot color aesthetics or handle detailed configurations 
    #   because its actions are strictly locked to the rigid parameters of the plot_data_tool.
    # - CodeAgent produced and compiled an independent Python code snippet, loaded the active data context, 
    #   and successfully generated an image file utilizing custom matplotlib arguments.
    # Did the ToolCallingAgent change the dot color? Did the CodeAgent?
    # - The ToolCallingAgent did not change the color because our pre-compiled tool definition does not 
    #   contain an argument for chart aesthetics. The CodeAgent successfully applied 'color="green"' via code generation.
    # What does this reveal about when each type of agent is more useful?
    # - ToolCallingAgents are ideal for rigid, transactional tasks where safety, fixed schemas, and exact 
    #   parameter tracking are required. CodeAgents are vastly superior for exploratory analysis, dynamic data scaling, 
    #   and complex customization workflows since they write structural workarounds natively.

# -Q9
# Reflection Comment Block

# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent.
#    A strict real-time banking system handling customer ATM withdrawals or wire transfers.

# 2. What property of the task makes it a good fit for a tool-based approach?
#    The strict demand for absolute determinism, predictable validation steps, and complete security. 
#    The system must never allow arbitrary code generation, class re-allocation, or dynamic loop manipulations; 
#    it should only accept specific arguments passed directly into predefined parameter schemas.

# 3. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent?
#    The execution of arbitrary, unbounded code strings inside local systems. Because CodeAgents parse model text 
#    and execute strings dynamically via an active interpreter sandbox, a structural hallucination or prompt injection 
#    could trigger infinite while-loops, rapid memory exhaustion, or accidental file system mutations. A ToolCallingAgent 
#    is completely safe from this risk since it can only access the exact functional parameters exposed inside its JSON schemas.

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    run_lesson02_q1()
    run_lesson02_q2()
    run_lesson02_q3()
    run_lesson03_q5_q6()
    run_lesson04_q7()
    run_lesson04_q8()
