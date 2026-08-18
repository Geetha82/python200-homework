import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import matplotlib
# Enforce a non-interactive backend to ensure plotting commands do not block execution threads
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

# Load global environmental token contexts
load_dotenv()
client = OpenAI()

# Defensive Engineering: Pre-populate the lesson's mock data file if missing to prevent file-not-found crashes
RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)
bike_csv_path = RESOURCES_DIR / "bike_commute.csv"
if not bike_csv_path.exists():
    mock_data = {
        "avg_traffic_density": [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.85, 0.3],
        "avg_speed_kmh": [22.5, 25.0, 19.8, 28.2, 23.1, 26.4, 20.5, 30.1],
        "avg_heart_rate": [145, 138, 152, 125, 140, 132, 148, 120],
        "duration_min": [45, 35, 50, 25, 40, 30, 48, 22]
    }
    pd.DataFrame(mock_data).to_csv(bike_csv_path, index=False)

# =====================================================================
# --- Lesson 02 ---
# =====================================================================

# --- Q1 ---
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Write the exact JSON schema dictionary using a literal flat configuration map matching the lesson style
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

print("--- Q1: Direct Function Call Testing ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# --- Q2 ---
def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Pure single-tool baseline array setup from the lesson
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
        print("[Q2 Agent Log]: Agentic mode engaged...")
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
        return second_response.choices[0].message.content or ''
    else:
        print("[Q2 Agent Log]: No tools needed....")
        return first_message.content or ''

# --- PREDICTION COMMENT BLOCK FOR Q2 ---
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
#    - No, it will not trigger a tool call. The only tool currently exposed to the model is 'get_current_time'.
#      Since checking the time has no logical relevance to conversion calculations, the model will ignore the tool.
# 2. How many API calls will be made to answer this query?
#    - Exactly 1 API call will be made. Because no tool call is requested by the model during the initial REASON step,
#      the execution path drops directly into the 'else' block, bypasses the secondary turn, and returns the response immediately.

print("\n--- Q2: Single-Tool Agent Execution ---")
q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result:", q2_result)


# --- Q3 ---
# Multi-tool extended version of the tools array configuration list
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
    },
    {
        'type': 'function',
        'function': celsius_to_fahrenheit_schema
    }
]

def run_agent_extended(user_prompt: str) -> str:
    """Extended ReAct-style agent supporting both get_current_time and celsius_to_fahrenheit tools."""
    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time and convert temperatures. Use the available tools whenever a user prompt requires them."""
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
        print("[Q3 Agent Log]: Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit(args.get("celsius"))
            else:
                tool_result = f'Error: unknown tool {function_name}.'
                
            print('Tool called:', function_name, "with args:", args)
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
        print("[Q3 Agent Log]: No tools needed....")
        return first_message.content or ''

print("\n--- Q3: Extended Multi-Tool Agent Execution ---")
response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Explanation Comment: A tool WAS called here ('celsius_to_fahrenheit') because the user query explicitly requested
# a temperature metric unit conversion, matching the semantic purpose and parameter criteria of the schema.

print()
response_b = run_agent_extended("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Explanation Comment: NO tool was called here because explaining the boiling point of water is a general, static 
# fact. The model handles this using its core parametric weights natively without triggering calculation functions.


# =====================================================================
# --- Lesson 03 ---
# =====================================================================

# --- Q4 ---
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
        title_csv = self.csv_name or "current CSV"
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.close()
            return f"Plotted {y} vs row index as a line plot."
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.close()
        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.
        
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error: return error
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"Columns not found. Current fields: {self.df.columns.tolist()}"}
        try:
            clean_df = self.df[[col1, col2]].dropna()
            r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[clean_df])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Failed to compute metrics: {str(e)}"}

# Verbatim literal lesson-style tools_schema declaration list array
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
                    "filename": {"type": "string", "description": "The name of the CSV file."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return column names for the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Return basic summary stats for one or more columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Simple summary for a single column using pandas.describe().",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data vectors from the active CSV dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string"},
                    "x": {"type": "string"},
                    "plot_type": {"type": "string"}
                },
                "required": ["y"]
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
                    "col1": {"type": "string", "description": "The first column name string."},
                    "col2": {"type": "string", "description": "The second column name string."}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# Instantiate single manager reference object
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

# Explicit layout entry map mapping function strings to method references
node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation
}


# --- Q5 ---
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """Run through one react-agent loop using a simple tool-using agent."""
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
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            messages.append(observe_tool_result(tool_call.id, result))
            
    return "I hit the tool-round limit. Try a simpler request."

print("\n--- Q5: Recreating Lesson 03 Scenario With New Correlation Tool ---")
SYSTEM_PROMPT_L3 = (
    "You are a helpful data assistant with access to tools for managing and analyzing CSV files. "
    "Use the available tools to satisfy the user's request. Always load a CSV file first before performing analysis."
)
messages = [{"role": "system", "content": SYSTEM_PROMPT_L3}]
q5_result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print("Final Response:", q5_result)


# --- Q6 ---
# --- ROLE EXPLANATION COMMENT MATRIX BLOCK ---
# - 'system': Sets global behavior constraints, operational identities, and foundational instructions for the agent lifecycle.
# - 'user': Captures explicit user text commands and inputs initiating tasks inside the ReAct pipeline.
# - 'assistant': Records the model's structural thought logs, text answers, or dynamic requests to invoke tools.
# - 'tool': Caches real data output responses generated by local functions, returning observations to the model's memory.
print("\n--- Q6: Serialized Message Tracking Logs Trace Dump ---")
print(json.dumps(messages, indent=2, default=str))


# =====================================================================
# --- Lesson 04 ---
# =====================================================================

# --- Q7 ---
@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: The name of the first column string.
        col2: The name of the second column string.

    Returns:
        dict: A dictionary containing the col1, col2, pearson_r, and p_value metrics.
    """
    return csv_manager.compute_correlation(col1, col2)

print("\n--- Q7: Smolagents Auto-Generated Tool Profiler Test ---")
print("Auto-Generated Description:", compute_correlation_tool.description)

# Comparison Comment:
# smolagents auto-generates the complete structural JSON argument schema by parsing the native Python function 
# signature definitions, standard parameter type hints, and docstring layout blocks directly. To produce an effective 
# description, the developer must provide explicit typing declarations (e.g., ': str') and high-quality docstrings 
# defining exact variable usages ('Args:') and explicit output returns ('Returns:').


# --- Q8 ---
# Build atomic tool mappings for smolagents framework tracking lists with mandatory docstring argument tags
@tool
def list_csv_files_tool() -> dict:
    """List available CSV files in resources/.

    Returns:
        dict: A lookup index layout tracking files on storage.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv_tool(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: The character string name of the target CSV file.

    Returns:
        dict: Completion status matrix message summaries.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns_tool() -> list:
    """Return column names for the currently loaded CSV.

    Returns:
        list: Collection array strings representing all headers.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns_tool(columns: list[str] = None) -> dict:
    """Return basic summary stats for one or more columns.

    Args:
        columns: Targeted specific subsets to profile or evaluate.

    Returns:
        dict: Multi-field matrix summarizing descriptive metrics.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column_tool(column: str) -> dict:
    """Simple summary for a single column using pandas.describe().

    Args:
        column: Single identity heading to generate descriptive metrics for.

    Returns:
        dict: Dictionary dataset containing target profiles.
    """
    return csv_manager.describe_column(column)

@tool
def plot_data_tool(y: str, x: str = None, plot_type: str = "line") -> str:
    """Plot from the active CSV dataset.

    Args:
        y: Main metric dependent vector drawn on the vertical axis.
        x: Optional control descriptor parameter on horizontal scale.
        plot_type: Formatting type style key, e.g. line or scatter.

    Returns:
        str: Diagnostic validation text receipt.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

TOOLS = [list_csv_files_tool, load_csv_tool, get_columns_tool, summarize_columns_tool, describe_column_tool, plot_data_tool, compute_correlation_tool]

model = OpenAIServerModel(api_key=os.getenv("OPENAI_API_KEY"), model_id="gpt-4o-mini")

tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT_L3)
code_agent = CodeAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT_L3, additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"], max_steps=8)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- Q8: Running ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("Tool Agent Response:", response_tool)

print("\n--- Q8: Running CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("Code Agent Response:", response_code)

# --- Q8 Behavioral Analysis Reflection Comment ---
# - What did each agent actually produce?
#   The ToolCallingAgent made text calls to 'load_csv_tool' and 'plot_data_tool', resulting in a standard, default-styled 
#   plot execution. The CodeAgent wrote a custom inline pandas and matplotlib python script scratchpad to load rows and 
#   render a refined visualization to disk.
# - Did the ToolCallingAgent change the dot color? Did the CodeAgent?
#   No, the ToolCallingAgent completely ignored the 'green dots' color request because its pre-built tool schema mapping 
#   lacked parameters to forward style overrides. The CodeAgent successfully parsed the rule, adding 'c="green"' and 
#   'marker="o"' directly inside its generated 'plt.scatter()' function statement.
# - What does this reveal about when each type of agent is more useful?
#   ToolCallingAgents excel at structured, rigid, predictable operations (like API CRUD tracking pipelines). CodeAgents 
#   are vastly superior for open-ended, iterative analytic tasks (like dynamic plotting, mathematical grouping, and custom ETL data cleaning).


# --- Q9 ---
# --- final assignment reflection answer block ---
# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent.
#    What property of the task makes it a good fit for a tool-based approach?
#    - Processing standard financial wire transfers or executing transactional database operations. 
#    - The core property making this a good fit is the absolute requirement for predictability, zero variance, and 
#      strict adherence to specific input schemas. It prevents the model from generating rogue execution steps.
# 2. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent?
#    - Runtime evaluation vulnerabilities and infinite consumption loops. Because a CodeAgent generates and executes 
#      arbitrary code strings dynamically inside an active environment, an unhandled calculation loop can exhaust 
#      compute resources or run destructive structural logic blocks that a strict, closed schema cannot trigger.
